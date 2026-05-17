"""Managed reference voice files for the OpenAI-compatible API."""

from __future__ import annotations

import errno
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VOICE_FILE_SUFFIX = ".wav"


@dataclass(frozen=True)
class VoiceFile:
    voice_id: str
    path: Path

    def metadata(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "id": self.voice_id,
            "object": "voice_file",
            "filename": self.path.name,
            "bytes": stat.st_size,
            "created_at": int(stat.st_mtime),
        }


class VoiceRegistry:
    """A small WAV-only voice registry rooted inside one configured directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()

    def status_metadata(self) -> dict[str, Any]:
        root = self.root
        files_error: str | None = None
        try:
            files = len(self.list_files())
        except OSError as exc:
            files = 0
            files_error = str(exc)
        metadata = {
            "dir": str(root),
            "dir_exists": root.is_dir(),
            "files": files,
            "formats": [VOICE_FILE_SUFFIX],
        }
        if files_error is not None:
            metadata["files_error"] = files_error
        return metadata

    def list_files(self) -> list[VoiceFile]:
        root = self._existing_root()
        if root is None:
            return []
        return [
            VoiceFile(voice_id=path.stem, path=path)
            for path in sorted(root.iterdir(), key=lambda item: item.name)
            if not path.is_symlink()
            and path.is_file()
            and path.suffix == VOICE_FILE_SUFFIX
            and self.is_managed_voice_id(path.stem)
        ]

    def get_file(self, voice_id: str) -> VoiceFile | None:
        self.validate_voice_id(voice_id)
        path = self._path_for_existing_root(voice_id)
        if path.is_symlink() or not path.is_file():
            return None
        return VoiceFile(voice_id=voice_id, path=path)

    def write_file(self, *, voice_id: str, filename: str, data: bytes, replace: bool) -> VoiceFile:
        self.validate_voice_id(voice_id)
        if Path(filename).suffix.lower() != VOICE_FILE_SUFFIX:
            raise ValueError("Managed reference voices must be uploaded as .wav files.")
        if not data:
            raise ValueError("Voice file must not be empty.")

        path = self._path_for(voice_id)
        if replace:
            self._replace_file(path, data)
        else:
            self._create_file(path, data, voice_id=voice_id)
        return VoiceFile(voice_id=voice_id, path=path)

    def delete_file(self, voice_id: str) -> bool:
        existing = self.get_file(voice_id)
        if existing is None:
            return False
        existing.path.unlink()
        return True

    def ensure_dir(self) -> Path:
        root = self.root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except FileExistsError as exc:
            raise NotADirectoryError(
                f"Configured voices directory is not a directory: {root}"
            ) from exc
        return root

    def _create_file(self, path: Path, data: bytes, *, voice_id: str) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(data)
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Voice {voice_id!r} already exists. Use PUT to replace it."
            ) from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("Managed reference voice files must not be symbolic links.") from exc
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _replace_file(self, path: Path, data: bytes) -> None:
        if path.is_symlink():
            raise ValueError("Managed reference voice files must not be symbolic links.")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(data)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _path_for(self, voice_id: str) -> Path:
        root = self.ensure_dir().resolve(strict=False)
        path = root / f"{voice_id}{VOICE_FILE_SUFFIX}"
        if path.parent != root:
            raise ValueError("voice_id must resolve inside the configured voices directory.")
        return path

    def _path_for_existing_root(self, voice_id: str) -> Path:
        root = self._existing_root()
        if root is None:
            root = self.root.resolve(strict=False)
        path = root / f"{voice_id}{VOICE_FILE_SUFFIX}"
        if path.parent != root:
            raise ValueError("voice_id must resolve inside the configured voices directory.")
        return path

    def _existing_root(self) -> Path | None:
        root = self.root
        if not root.exists():
            return None
        if not root.is_dir():
            raise NotADirectoryError(f"Configured voices directory is not a directory: {root}")
        return root.resolve(strict=False)

    @staticmethod
    def validate_voice_id(voice_id: str) -> None:
        if not VoiceRegistry.is_managed_voice_id(voice_id):
            raise ValueError(
                "voice_id must contain only ASCII letters, numbers, underscores, or hyphens."
            )

    @staticmethod
    def is_managed_voice_id(voice_id: str) -> bool:
        return bool(voice_id and VOICE_ID_PATTERN.fullmatch(voice_id) is not None)

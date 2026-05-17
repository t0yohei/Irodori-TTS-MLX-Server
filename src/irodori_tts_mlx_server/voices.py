"""Managed reference voice files for the OpenAI-compatible API."""

from __future__ import annotations

import re
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
        return {
            "dir": str(root),
            "dir_exists": root.is_dir(),
            "files": len(self.list_files()) if root.is_dir() else 0,
            "formats": [VOICE_FILE_SUFFIX],
        }

    def list_files(self) -> list[VoiceFile]:
        root = self.root
        if not root.is_dir():
            return []
        return [
            VoiceFile(voice_id=path.stem, path=path)
            for path in sorted(root.iterdir(), key=lambda item: item.name)
            if path.is_file()
            and path.suffix.lower() == VOICE_FILE_SUFFIX
            and self.is_managed_voice_id(path.stem)
        ]

    def get_file(self, voice_id: str) -> VoiceFile | None:
        self.validate_voice_id(voice_id)
        path = self._path_for_existing_root(voice_id)
        if not path.is_file():
            return None
        return VoiceFile(voice_id=voice_id, path=path)

    def write_file(self, *, voice_id: str, filename: str, data: bytes, replace: bool) -> VoiceFile:
        self.validate_voice_id(voice_id)
        if Path(filename).suffix.lower() != VOICE_FILE_SUFFIX:
            raise ValueError("Managed reference voices must be uploaded as .wav files.")
        if not data:
            raise ValueError("Voice file must not be empty.")

        path = self._path_for(voice_id)
        if path.exists() and not replace:
            raise FileExistsError(f"Voice {voice_id!r} already exists. Use PUT to replace it.")
        path.write_bytes(data)
        return VoiceFile(voice_id=voice_id, path=path)

    def delete_file(self, voice_id: str) -> bool:
        existing = self.get_file(voice_id)
        if existing is None:
            return False
        existing.path.unlink()
        return True

    def ensure_dir(self) -> Path:
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _path_for(self, voice_id: str) -> Path:
        root = self.ensure_dir().resolve(strict=False)
        path = (root / f"{voice_id}{VOICE_FILE_SUFFIX}").resolve(strict=False)
        if path.parent != root:
            raise ValueError("voice_id must resolve inside the configured voices directory.")
        return path

    def _path_for_existing_root(self, voice_id: str) -> Path:
        root = self.root.resolve(strict=False)
        path = (root / f"{voice_id}{VOICE_FILE_SUFFIX}").resolve(strict=False)
        if path.parent != root:
            raise ValueError("voice_id must resolve inside the configured voices directory.")
        return path

    @staticmethod
    def validate_voice_id(voice_id: str) -> None:
        if not VoiceRegistry.is_managed_voice_id(voice_id):
            raise ValueError(
                "voice_id must contain only ASCII letters, numbers, underscores, or hyphens."
            )

    @staticmethod
    def is_managed_voice_id(voice_id: str) -> bool:
        return bool(voice_id and VOICE_ID_PATTERN.fullmatch(voice_id) is not None)

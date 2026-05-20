"""Managed reference voice files for the OpenAI-compatible API."""

from __future__ import annotations

import errno
import os
import re
import stat as stat_module
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VOICE_FILE_SUFFIX = ".wav"
VOICE_FILE_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm")
LATENT_FILE_SUFFIXES = (".pt", ".pth")
STALE_CREATE_LOCK_SECONDS = 300
HARD_LINK_UNSUPPORTED_ERRNOS = {errno.EPERM, errno.EXDEV} | {
    code
    for code in (getattr(errno, "ENOTSUP", None), getattr(errno, "EOPNOTSUPP", None))
    if code is not None
}


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
    """A small managed voice registry rooted inside one configured directory."""

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
            "formats": list(VOICE_FILE_SUFFIXES),
        }
        if files_error is not None:
            metadata["files_error"] = files_error
        return metadata

    def list_files(self) -> list[VoiceFile]:
        root = self._existing_root()
        if root is None:
            return []
        voice_ids: set[str] = set()
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix not in VOICE_FILE_SUFFIXES
                or not self.is_managed_voice_id(path.stem)
            ):
                continue
            voice_ids.add(path.stem)
        return [
            voice_file
            for voice_id in sorted(voice_ids)
            if (voice_file := self.get_file(voice_id)) is not None
        ]

    def get_file(self, voice_id: str) -> VoiceFile | None:
        self.validate_voice_id(voice_id)
        root = self._existing_root()
        if root is None:
            return None
        for path in self._candidate_paths(root, voice_id):
            if path.is_symlink() or not path.is_file():
                continue
            return VoiceFile(voice_id=voice_id, path=path)
        return None

    def write_file(self, *, voice_id: str, filename: str, data: bytes, replace: bool) -> VoiceFile:
        self.validate_voice_id(voice_id)
        suffix = Path(filename).suffix.lower()
        if suffix not in VOICE_FILE_SUFFIXES:
            allowed = ", ".join(VOICE_FILE_SUFFIXES)
            raise ValueError(f"Managed reference voices must use one of: {allowed}.")
        if not data:
            raise ValueError("Voice file must not be empty.")
        if not replace and self.get_file(voice_id) is not None:
            raise FileExistsError(
                f"Voice {voice_id!r} already exists. Use PUT to replace it."
            )

        path = self._path_for(voice_id, suffix=suffix)
        if replace:
            self._replace_file(path, data, voice_id=voice_id)
        else:
            self._create_unique_file(path, data, voice_id=voice_id)
        return VoiceFile(voice_id=voice_id, path=path)

    def delete_file(self, voice_id: str) -> bool:
        existing = self.get_file(voice_id)
        if existing is None:
            return False
        for candidate in self._candidate_paths(existing.path.parent, voice_id):
            if candidate.is_file():
                candidate.unlink()
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
            if exc.errno in HARD_LINK_UNSUPPORTED_ERRNOS:
                self._create_file_without_hard_link(path, data, voice_id=voice_id)
                return
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _create_file_without_hard_link(self, path: Path, data: bytes, *, voice_id: str) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd: int | None = None
        created = False
        write_failed = False
        try:
            fd = os.open(path, flags, 0o600)
            created = True
            with os.fdopen(fd, "wb") as output_file:
                fd = None
                output_file.write(data)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Voice {voice_id!r} already exists. Use PUT to replace it."
            ) from exc
        except OSError as exc:
            write_failed = True
            if exc.errno == errno.ELOOP:
                raise ValueError("Managed reference voice files must not be symbolic links.") from exc
            raise
        finally:
            if fd is not None:
                os.close(fd)
            if created:
                try:
                    if write_failed or path.stat().st_size != len(data):
                        path.unlink(missing_ok=True)
                except OSError:
                    path.unlink(missing_ok=True)
                    raise

    def _create_unique_file(self, path: Path, data: bytes, *, voice_id: str) -> None:
        lock_path = path.parent / f".{voice_id}.create.lock"
        self._create_create_lock(lock_path, voice_id=voice_id)
        try:
            if self.get_file(voice_id) is not None:
                raise FileExistsError(
                    f"Voice {voice_id!r} already exists. Use PUT to replace it."
                )
            self._create_file(path, data, voice_id=voice_id)
        finally:
            lock_path.unlink(missing_ok=True)

    def _create_create_lock(self, lock_path: Path, *, voice_id: str) -> None:
        try:
            self._create_file(lock_path, b"", voice_id=voice_id)
            return
        except FileExistsError:
            if self.get_file(voice_id) is not None:
                raise
            try:
                stat = lock_path.lstat()
            except FileNotFoundError:
                self._create_file(lock_path, b"", voice_id=voice_id)
                return
            age_seconds = time.time() - stat.st_mtime
            if not stat_module.S_ISREG(stat.st_mode) or age_seconds < STALE_CREATE_LOCK_SECONDS:
                raise
            lock_path.unlink(missing_ok=True)
        self._create_file(lock_path, b"", voice_id=voice_id)

    def _replace_file(self, path: Path, data: bytes, *, voice_id: str) -> None:
        for candidate in self._candidate_paths(path.parent, voice_id):
            if candidate.is_symlink():
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
            for candidate in self._candidate_paths(path.parent, voice_id):
                if candidate != path and candidate.is_file():
                    candidate.unlink()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def validate_reference_path(self, value: str) -> Path:
        parsed = urlparse(value)
        if parsed.scheme or parsed.netloc:
            raise ValueError("irodori.ref_wav must not be a remote URL.")

        root = self._existing_root()
        if root is None:
            root = self.root.resolve(strict=False)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_symlink():
            raise ValueError("irodori.ref_wav must not be a symbolic link.")
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ValueError(
                "irodori.ref_wav must resolve inside the configured voices directory."
            )
        if resolved.suffix.lower() not in VOICE_FILE_SUFFIXES:
            allowed = ", ".join(VOICE_FILE_SUFFIXES)
            raise ValueError(f"irodori.ref_wav must use one of: {allowed}.")
        if resolved.parent != root or not self.is_managed_voice_id(resolved.stem):
            raise ValueError("irodori.ref_wav must refer to a managed voice file.")
        if resolved not in self._candidate_paths(root, resolved.stem):
            raise ValueError("irodori.ref_wav must refer to a managed voice file.")
        if not resolved.is_file():
            raise ValueError("irodori.ref_wav must refer to a managed voice file.")
        return resolved

    def _path_for(self, voice_id: str, *, suffix: str) -> Path:
        root = self.ensure_dir().resolve(strict=False)
        path = root / f"{voice_id}{suffix}"
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

    def _candidate_paths(self, root: Path, voice_id: str) -> list[Path]:
        return [root / f"{voice_id}{suffix}" for suffix in VOICE_FILE_SUFFIXES]

    @staticmethod
    def validate_voice_id(voice_id: str) -> None:
        if not VoiceRegistry.is_managed_voice_id(voice_id):
            raise ValueError(
                "voice_id must contain only ASCII letters, numbers, underscores, or hyphens."
            )

    @staticmethod
    def is_managed_voice_id(voice_id: str) -> bool:
        return bool(voice_id and VOICE_ID_PATTERN.fullmatch(voice_id) is not None)

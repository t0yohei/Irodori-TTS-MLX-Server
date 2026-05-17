"""Server configuration loaded from local environment variables."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    """FastAPI server controls that do not belong to the MLX runtime adapter."""

    bearer_token: str | None = None
    max_concurrent_synthesis: int = 1
    queue_timeout_seconds: float = 30.0
    voices_dir: Path = Path("voices")
    max_voice_upload_bytes: int = 50 * 1024 * 1024

    @property
    def auth_enabled(self) -> bool:
        return bool(self.bearer_token)


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_float(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(parsed_value):
        raise ValueError(f"{name} must be a finite number.")
    return parsed_value


def server_config_from_env() -> ServerConfig:
    token = os.getenv("IRODORI_SERVER_BEARER_TOKEN") or os.getenv("IRODORI_API_KEY")
    max_concurrent_synthesis = _env_int("IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS", default=1)
    queue_timeout_seconds = _env_float("IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS", default=30.0)
    voices_dir = Path(os.getenv("IRODORI_SERVER_VOICES_DIR") or "voices")
    max_voice_upload_bytes = _env_int(
        "IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES", default=50 * 1024 * 1024
    )
    if max_concurrent_synthesis < 1:
        raise ValueError("IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS must be >= 1.")
    if queue_timeout_seconds < 0:
        raise ValueError("IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS must be >= 0.")
    if max_voice_upload_bytes < 1:
        raise ValueError("IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES must be >= 1.")
    return ServerConfig(
        bearer_token=token,
        max_concurrent_synthesis=max_concurrent_synthesis,
        queue_timeout_seconds=queue_timeout_seconds,
        voices_dir=voices_dir,
        max_voice_upload_bytes=max_voice_upload_bytes,
    )

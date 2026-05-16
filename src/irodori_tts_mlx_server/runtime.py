"""Runtime boundary for speech generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SpeechGenerationRequest:
    model: str
    input: str
    voice: str
    response_format: str
    speed: float
    irodori: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechGenerationResult:
    audio: bytes
    media_type: str


class RuntimeUnavailableError(RuntimeError):
    """Raised when real model weights are not configured for local generation."""


class SpeechRuntime(Protocol):
    def list_models(self) -> list[str]:
        """Return OpenAI-compatible model identifiers served by this runtime."""

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        """Generate complete audio bytes for a validated speech request."""


class UnconfiguredSpeechRuntime:
    """Import-safe runtime placeholder used until real Irodori weights are configured."""

    def list_models(self) -> list[str]:
        return ["irodori-tts-mlx"]

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeUnavailableError("Irodori-TTS-MLX runtime is not configured.")

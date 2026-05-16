"""FastAPI application bootstrap."""

from __future__ import annotations

from irodori_tts_mlx_server.factory import AudioSpeechRequest, create_app, openai_error

app = create_app()

__all__ = ["AudioSpeechRequest", "app", "create_app", "openai_error"]

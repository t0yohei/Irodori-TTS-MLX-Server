"""Irodori-TTS-MLX FastAPI server package."""

from __future__ import annotations

from typing import Any

from irodori_tts_mlx_server.factory import create_app

__all__ = ["app", "create_app"]


class LazyDefaultApp:
    """ASGI proxy that avoids constructing the default app during package import."""

    def __init__(self) -> None:
        self._app: Any | None = None

    def _get_app(self) -> Any:
        if self._app is None:
            self._app = create_app()
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_app(), name)


app = LazyDefaultApp()

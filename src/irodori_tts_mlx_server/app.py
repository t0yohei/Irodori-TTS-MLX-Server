"""FastAPI application bootstrap."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from irodori_tts_mlx_server.runtime import (
    RuntimeUnavailableError,
    SpeechGenerationRequest,
    SpeechRuntime,
    UnconfiguredSpeechRuntime,
)


def openai_error(
    message: str,
    *,
    status_code: int,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


class AudioSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    input: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    irodori: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


def create_app(runtime: SpeechRuntime | None = None) -> FastAPI:
    speech_runtime = runtime or UnconfiguredSpeechRuntime()
    app = FastAPI(title="Irodori-TTS-MLX Server", version="0.1.0")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = first_error.get("loc", ())
        param = ".".join(str(part) for part in location if part != "body") or None
        message = first_error.get("msg", "Invalid request.")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": "validation_error",
                }
            },
        )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", tags=["openai"])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "irodori-tts-mlx",
                }
                for model_id in speech_runtime.list_models()
            ],
        }

    @app.post("/v1/audio/speech", tags=["openai"])
    async def create_speech(api_request: Request, request: AudioSpeechRequest) -> Response:
        if request.stream or "text/event-stream" in api_request.headers.get("accept", ""):
            raise openai_error(
                "Streaming audio responses and SSE are not supported; request complete audio bytes.",
                status_code=400,
                param="stream",
                code="unsupported_streaming",
            )
        if request.model not in speech_runtime.list_models():
            raise openai_error(
                f"Model '{request.model}' is not available.",
                status_code=404,
                param="model",
                code="model_not_found",
            )

        generation_request = SpeechGenerationRequest(
            model=request.model,
            input=request.input,
            voice=request.voice,
            response_format=request.response_format,
            speed=request.speed,
            irodori=request.irodori,
        )
        try:
            result = speech_runtime.generate_speech(generation_request)
        except RuntimeUnavailableError as exc:
            raise openai_error(
                str(exc),
                status_code=503,
                error_type="server_error",
                code="runtime_unavailable",
            ) from exc

        return Response(content=result.audio, media_type=result.media_type)

    return app


app = create_app()

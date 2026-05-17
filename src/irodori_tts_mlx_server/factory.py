"""FastAPI application factory without module-level app construction."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from irodori_tts_mlx_server.config import ServerConfig, server_config_from_env
from irodori_tts_mlx_server.audio import (
    AudioConversionError,
    convert_audio_response,
    ensure_response_format_available,
)
from irodori_tts_mlx_server.runtime import (
    RuntimeRequestError,
    RuntimeUnavailableError,
    SpeechGenerationRequest,
    SpeechRuntime,
    create_default_runtime,
)
from irodori_tts_mlx_server.voices import VoiceRegistry


class ServerConfigurationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


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
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    irodori: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    stream_format: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_compatibility_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        irodori = normalized.get("irodori")
        if irodori is None:
            irodori_options: dict[str, Any] = {}
        elif isinstance(irodori, dict):
            irodori_options = dict(irodori)
        else:
            return normalized

        for alias, canonical in TOP_LEVEL_IRODORI_ALIASES.items():
            if alias not in normalized:
                continue
            _set_irodori_option(
                irodori_options,
                canonical,
                normalized.pop(alias),
                alias=alias,
            )
        if "context_kv_cache" in normalized:
            _set_inverted_irodori_option(
                irodori_options,
                "no_context_kv_cache",
                normalized.pop("context_kv_cache"),
                alias="context_kv_cache",
            )

        _normalize_irodori_aliases(irodori_options)
        normalized["irodori"] = irodori_options
        return normalized


TOP_LEVEL_IRODORI_ALIASES = {
    "no_ref": "no_reference",
    "seconds": "seconds",
    "duration_scale": "duration_scale",
    "num_steps": "num_steps",
    "seed": "seed",
    "cfg_scale_text": "cfg_scale_text",
    "cfg_scale_caption": "cfg_scale_caption",
    "cfg_scale_speaker": "cfg_scale_speaker",
    "cfg_guidance_mode": "cfg_guidance_mode",
    "cfg_min_t": "cfg_min_t",
    "cfg_max_t": "cfg_max_t",
    "max_ref_seconds": "max_reference_seconds",
    "max_reference_seconds": "max_reference_seconds",
    "no_context_kv_cache": "no_context_kv_cache",
    "chunking": "chunking",
    "chunking_enabled": "chunking",
    "chunk_max_chars": "chunk_max_chars",
}

IRODORI_OPTION_ALIASES = {
    "ref_wav": "reference_wav",
    "no_ref": "no_reference",
    "max_ref_seconds": "max_reference_seconds",
    "chunking_enabled": "chunking",
}

UNSUPPORTED_IRODORI_OPTIONS = {
    "ref_latent",
    "lora_adapter",
    "chunk_min_chars",
    "min_seconds",
    "max_seconds",
    "ref_normalize_db",
    "ref_ensure_max",
    "t_schedule_mode",
    "sway_coeff",
    "num_candidates",
    "decode_mode",
    "cfg_scale",
    "truncation_factor",
    "rescale_k",
    "rescale_sigma",
    "speaker_kv_scale",
    "speaker_kv_min_t",
    "speaker_kv_max_layers",
    "trim_tail",
    "tail_window_size",
    "tail_std_threshold",
    "tail_mean_threshold",
    "max_text_len",
}


def _set_irodori_option(options: dict[str, Any], canonical: str, value: Any, *, alias: str) -> None:
    if canonical in options and options[canonical] != value:
        raise ValueError(f"{alias} conflicts with irodori.{canonical}.")
    options[canonical] = value


def _set_inverted_irodori_option(
    options: dict[str, Any], canonical: str, value: Any, *, alias: str
) -> None:
    if isinstance(value, bool):
        inverted = not value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            inverted = False
        elif normalized in {"0", "false", "no", "off"}:
            inverted = True
        else:
            raise ValueError(f"irodori.{alias} must be a boolean.")
    else:
        raise ValueError(f"irodori.{alias} must be a boolean.")
    _set_irodori_option(options, canonical, inverted, alias=f"irodori.{alias}")


def _normalize_irodori_aliases(options: dict[str, Any]) -> None:
    unsupported = sorted(UNSUPPORTED_IRODORI_OPTIONS.intersection(options))
    if unsupported:
        names = ", ".join(f"irodori.{name}" for name in unsupported)
        raise ValueError(f"Unsupported upstream Irodori option(s): {names}.")

    if "context_kv_cache" in options:
        _set_inverted_irodori_option(
            options,
            "no_context_kv_cache",
            options.pop("context_kv_cache"),
            alias="context_kv_cache",
        )
    for alias, canonical in IRODORI_OPTION_ALIASES.items():
        if alias not in options:
            continue
        _set_irodori_option(
            options,
            canonical,
            options.pop(alias),
            alias=f"irodori.{alias}",
        )


def _bool_like_option(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


class VoiceUploadResponse(BaseModel):
    id: str
    object: Literal["voice_file"]
    filename: str
    bytes: int
    created_at: int


class SynthesisLimiter:
    def __init__(self, *, max_concurrent: int, queue_timeout_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_timeout_seconds = queue_timeout_seconds

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            if self._queue_timeout_seconds == 0:
                if self._semaphore.locked():
                    raise TimeoutError
                await self._semaphore.acquire()
            else:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=self._queue_timeout_seconds
                )
        except TimeoutError as exc:
            raise openai_error(
                "Synthesis queue is full or the model is still loading; retry later.",
                status_code=503,
                error_type="server_error",
                code="synthesis_queue_timeout",
            ) from exc
        try:
            yield
        finally:
            self._semaphore.release()


VOICE_UPLOAD_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _is_voice_upload_route(request: Request) -> bool:
    return request.method in {"POST", "PUT"} and (
        request.url.path == "/v1/audio/voices" or request.url.path.startswith("/v1/audio/voices/")
    )


def _voice_file_too_large_error(max_bytes: int) -> HTTPException:
    return openai_error(
        f"Voice file must be no larger than {max_bytes} bytes.",
        status_code=413,
        param="file",
        code="voice_file_too_large",
    )


def _voice_storage_error(exc: OSError) -> HTTPException:
    return openai_error(
        f"Managed reference voice storage is unavailable: {exc}",
        status_code=503,
        error_type="server_error",
        param="voice_id",
        code="voice_storage_unavailable",
    )


def _install_voice_upload_size_guard(config: ServerConfig, request: Request) -> None:
    if not _is_voice_upload_route(request):
        return
    max_request_bytes = config.max_voice_upload_bytes + VOICE_UPLOAD_MULTIPART_OVERHEAD_BYTES
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            request_bytes = int(content_length)
        except ValueError as exc:
            raise openai_error(
                "Invalid Content-Length header.",
                status_code=400,
                param="content-length",
                code="invalid_content_length",
            ) from exc
        if request_bytes > max_request_bytes:
            raise _voice_file_too_large_error(config.max_voice_upload_bytes)

    receive = request._receive
    received_bytes = 0

    async def limited_receive() -> Any:
        nonlocal received_bytes
        message = await receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"")
            if isinstance(body, bytes):
                received_bytes += len(body)
                if received_bytes > max_request_bytes:
                    raise _voice_file_too_large_error(config.max_voice_upload_bytes)
        return message

    request._receive = limited_receive


async def _read_voice_upload(file: UploadFile, *, max_bytes: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise _voice_file_too_large_error(max_bytes)
    return bytes(data)


def _server_configuration_error(message: str) -> HTTPException:
    return openai_error(
        message,
        status_code=503,
        error_type="server_error",
        code="server_configuration_error",
    )


def _resolve_server_config(config: ServerConfig | None) -> ServerConfig:
    if config is not None:
        return config
    try:
        return server_config_from_env()
    except ValueError as exc:
        raise ServerConfigurationError(str(exc)) from exc


def _server_health_metadata(
    config: ServerConfig,
    configuration_error: ServerConfigurationError | None,
    voice_registry: VoiceRegistry,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "auth_enabled": config.auth_enabled,
        "max_concurrent_synthesis": config.max_concurrent_synthesis,
        "queue_timeout_seconds": config.queue_timeout_seconds,
        "voices": voice_registry.status_metadata(),
    }
    if configuration_error is not None:
        metadata.update(
            {
                "status": "configuration_error",
                "error": {
                    "code": "server_configuration_error",
                    "message": configuration_error.message,
                },
            }
        )
    return metadata


def _require_bearer_auth(config: ServerConfig, request: Request) -> None:
    if not config.auth_enabled:
        return
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, config.bearer_token or ""):
        raise openai_error(
            "Missing or invalid bearer token.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )


def _authentication_error_response(exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


def _apply_managed_voice_reference(
    request: AudioSpeechRequest, voice_registry: VoiceRegistry
) -> dict[str, Any]:
    irodori_options = dict(request.irodori)
    if "reference_wav" in irodori_options:
        return irodori_options
    if "no_reference" in irodori_options:
        no_reference = _bool_like_option(irodori_options["no_reference"])
        if no_reference is not False:
            return irodori_options

    if not voice_registry.is_managed_voice_id(request.voice):
        return irodori_options

    try:
        voice_file = voice_registry.get_file(request.voice)
    except ValueError:
        return irodori_options
    except OSError:
        return irodori_options
    if voice_file is None:
        return irodori_options

    irodori_options["reference_wav"] = str(voice_file.path)
    irodori_options["no_reference"] = False
    return irodori_options


def create_app(runtime: SpeechRuntime | None = None, config: ServerConfig | None = None) -> FastAPI:
    speech_runtime = runtime if runtime is not None else create_default_runtime()
    server_configuration_error: ServerConfigurationError | None = None
    try:
        server_config = _resolve_server_config(config)
    except ServerConfigurationError as exc:
        server_configuration_error = exc
        server_config = ServerConfig()
    synthesis_limiter = SynthesisLimiter(
        max_concurrent=server_config.max_concurrent_synthesis,
        queue_timeout_seconds=server_config.queue_timeout_seconds,
    )
    voice_registry = VoiceRegistry(server_config.voices_dir)
    app = FastAPI(title="Irodori-TTS-MLX Server", version="0.1.0")

    @app.middleware("http")
    async def authenticate_openai_routes(api_request: Request, call_next: Any) -> Response:
        try:
            if api_request.url.path.startswith("/v1/"):
                _require_bearer_auth(server_config, api_request)
                _install_voice_upload_size_guard(server_config, api_request)
            return await call_next(api_request)
        except HTTPException as exc:
            return _authentication_error_response(exc)

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
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "speech_runtime": speech_runtime.status_metadata(),
            "server": _server_health_metadata(
                server_config, server_configuration_error, voice_registry
            ),
        }

    @app.get("/v1/models", tags=["openai"])
    async def list_models(api_request: Request) -> dict[str, Any]:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
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

    @app.get("/v1/audio/voices", tags=["openai"])
    async def list_voices(api_request: Request) -> dict[str, Any]:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
        try:
            voice_files = voice_registry.list_files()
        except OSError as exc:
            raise _voice_storage_error(exc)
        return {
            "object": "list",
            "data": [
                {
                    "id": voice_file.voice_id,
                    "object": "voice",
                    "ref_wav": str(voice_file.path),
                    "ref_latent": None,
                    "no_ref": False,
                }
                for voice_file in voice_files
            ],
        }

    @app.post("/v1/audio/voices", status_code=201, tags=["openai"])
    async def upload_voice(
        api_request: Request,
        file: UploadFile = File(...),
        voice_id: str | None = Form(default=None),
    ) -> VoiceUploadResponse:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
        filename = file.filename or ""
        resolved_voice_id = (voice_id or filename.rsplit(".", 1)[0]).strip()
        data = await _read_voice_upload(file, max_bytes=server_config.max_voice_upload_bytes)
        try:
            voice_file = voice_registry.write_file(
                voice_id=resolved_voice_id,
                filename=filename,
                data=data,
                replace=False,
            )
        except FileExistsError as exc:
            raise openai_error(str(exc), status_code=409, param="voice_id", code="voice_exists")
        except ValueError as exc:
            raise openai_error(str(exc), status_code=400, param="voice_id", code="invalid_voice")
        except OSError as exc:
            raise _voice_storage_error(exc)
        return VoiceUploadResponse(**voice_file.metadata())

    @app.get("/v1/audio/voices/{voice_id}", tags=["openai"])
    async def get_voice(api_request: Request, voice_id: str) -> VoiceUploadResponse:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
        try:
            voice_file = voice_registry.get_file(voice_id)
        except ValueError as exc:
            raise openai_error(str(exc), status_code=400, param="voice_id", code="invalid_voice")
        except OSError as exc:
            raise _voice_storage_error(exc)
        if voice_file is None:
            raise openai_error(
                f"Voice {voice_id!r} was not found.",
                status_code=404,
                param="voice_id",
                code="voice_not_found",
            )
        return VoiceUploadResponse(**voice_file.metadata())

    @app.put("/v1/audio/voices/{voice_id}", tags=["openai"])
    async def replace_voice(
        api_request: Request,
        voice_id: str,
        file: UploadFile = File(...),
    ) -> VoiceUploadResponse:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
        try:
            if voice_registry.get_file(voice_id) is None:
                raise openai_error(
                    f"Voice {voice_id!r} was not found.",
                    status_code=404,
                    param="voice_id",
                    code="voice_not_found",
                )
        except ValueError as exc:
            raise openai_error(str(exc), status_code=400, param="voice_id", code="invalid_voice")
        except OSError as exc:
            raise _voice_storage_error(exc)

        try:
            voice_file = voice_registry.write_file(
                voice_id=voice_id,
                filename=file.filename or "",
                data=await _read_voice_upload(file, max_bytes=server_config.max_voice_upload_bytes),
                replace=True,
            )
        except ValueError as exc:
            raise openai_error(str(exc), status_code=400, param="voice_id", code="invalid_voice")
        except OSError as exc:
            raise _voice_storage_error(exc)
        return VoiceUploadResponse(**voice_file.metadata())

    @app.delete("/v1/audio/voices/{voice_id}", tags=["openai"])
    async def delete_voice(api_request: Request, voice_id: str) -> dict[str, Any]:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
        try:
            deleted = voice_registry.delete_file(voice_id)
        except ValueError as exc:
            raise openai_error(str(exc), status_code=400, param="voice_id", code="invalid_voice")
        except OSError as exc:
            raise _voice_storage_error(exc)
        if not deleted:
            raise openai_error(
                f"Voice {voice_id!r} was not found.",
                status_code=404,
                param="voice_id",
                code="voice_not_found",
            )
        return {"id": voice_id, "object": "voice_file", "deleted": True}

    @app.post("/v1/audio/speech", tags=["openai"])
    async def create_speech(api_request: Request, request: AudioSpeechRequest) -> Response:
        if server_configuration_error is not None:
            raise _server_configuration_error(server_configuration_error.message)
        _require_bearer_auth(server_config, api_request)
        if (
            request.stream
            or request.stream_format is not None
            or "text/event-stream" in api_request.headers.get("accept", "")
        ):
            param = "stream_format" if request.stream_format is not None else "stream"
            raise openai_error(
                "Streaming audio responses and SSE are not supported; request complete audio bytes.",
                status_code=400,
                param=param,
                code="unsupported_streaming",
            )
        if request.model not in speech_runtime.list_models():
            raise openai_error(
                f"Model '{request.model}' is not available.",
                status_code=404,
                param="model",
                code="model_not_found",
            )
        try:
            ensure_response_format_available(request.response_format)
        except AudioConversionError as exc:
            status_code = 400 if exc.code == "unsupported_response_format" else 503
            raise openai_error(
                str(exc),
                status_code=status_code,
                error_type="server_error" if status_code >= 500 else "invalid_request_error",
                param="response_format",
                code=exc.code,
            ) from exc
        irodori_options = _apply_managed_voice_reference(request, voice_registry)
        generation_request = SpeechGenerationRequest(
            model=request.model,
            input=request.input,
            voice=request.voice,
            response_format="wav",
            speed=request.speed,
            irodori=irodori_options,
        )
        try:
            async with synthesis_limiter.slot():
                result = await run_in_threadpool(speech_runtime.generate_speech, generation_request)
        except RuntimeRequestError as exc:
            raise openai_error(
                str(exc),
                status_code=400,
                param="irodori",
                code="invalid_irodori_options",
            ) from exc
        except RuntimeUnavailableError as exc:
            raise openai_error(
                str(exc),
                status_code=503,
                error_type="server_error",
                code="runtime_unavailable",
            ) from exc
        try:
            converted_result = await run_in_threadpool(
                convert_audio_response, result, request.response_format
            )
        except AudioConversionError as exc:
            status_code = 400 if exc.code == "unsupported_response_format" else 503
            raise openai_error(
                str(exc),
                status_code=status_code,
                error_type="server_error" if status_code >= 500 else "invalid_request_error",
                param="response_format",
                code=exc.code,
            ) from exc

        return Response(content=converted_result.audio, media_type=converted_result.media_type)

    return app

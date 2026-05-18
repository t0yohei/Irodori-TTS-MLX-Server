"""Runtime boundary for speech generation."""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
import threading
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

PRESET_NUM_STEPS = {
    "fast": 12,
    "balanced": 24,
    "quality": 40,
}
DEFAULT_TAIL_TRIM_MS = 0
DEFAULT_TAIL_SILENCE_TRIM_MS = 0
DEFAULT_TAIL_SILENCE_KEEP_MS = 40
DEFAULT_TAIL_SILENCE_THRESHOLD = 256
WAV_MEDIA_TYPE = "audio/wav"
LORA_ADAPTER_OPTION = "lora_adapter"
logger = logging.getLogger("irodori_tts_mlx_server.runtime")


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


class RuntimeRequestError(ValueError):
    """Raised when server request options cannot be mapped to the MLX runtime."""


class SpeechRuntime(Protocol):
    def list_models(self) -> list[str]:
        """Return OpenAI-compatible model identifiers served by this runtime."""

    def status_metadata(self) -> dict[str, Any]:
        """Return health metadata for this runtime without forcing a model load."""

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        """Generate complete audio bytes for a validated speech request."""


class UnconfiguredSpeechRuntime:
    """Import-safe runtime placeholder used until real Irodori weights are configured."""

    def list_models(self) -> list[str]:
        return ["irodori-tts-mlx"]

    def status_metadata(self) -> dict[str, Any]:
        return {
            "runtime": "unconfigured",
            "configured": False,
            "loaded": False,
            "load_state": "unconfigured",
            "model_id": "irodori-tts-mlx",
        }

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeUnavailableError(
            "Irodori-TTS-MLX runtime is not configured. Set IRODORI_MLX_WEIGHTS_DIR "
            "or IRODORI_MLX_WEIGHTS_REPO for a hosted converted weights layout."
        )


class InvalidConfigurationSpeechRuntime:
    """Import-safe runtime placeholder used when environment configuration is invalid."""

    def __init__(self, message: str, *, model_id: str = "irodori-tts-mlx") -> None:
        self.message = message
        self.model_id = model_id

    def list_models(self) -> list[str]:
        return [self.model_id]

    def status_metadata(self) -> dict[str, Any]:
        return {
            "runtime": "configuration_error",
            "configured": False,
            "loaded": False,
            "load_state": "failed",
            "model_id": self.model_id,
            "last_load_error": self.message,
        }

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeUnavailableError(self.message)


@dataclass(frozen=True)
class IrodoriRuntimeConfig:
    """Configuration for the real Irodori-TTS-MLX runtime adapter."""

    model_id: str = "irodori-tts-mlx"
    weights_dir: str | None = None
    weights_repo: str | None = None
    weights_revision: str | None = None
    text_tokenizer_repo: str | None = None
    caption_tokenizer_repo: str | None = None
    text_max_length: int = 256
    caption_max_length: int | None = None
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    codec_device: str = "cpu"
    codec_runtime_mode: str = "persistent"
    enable_watermark: bool = False
    normalize_codec_audio: bool = True
    preload: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.weights_dir or self.weights_repo)


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeUnavailableError(f"{name} must be an integer.") from exc


def runtime_config_from_env() -> IrodoriRuntimeConfig:
    return IrodoriRuntimeConfig(
        model_id=os.getenv("IRODORI_MLX_MODEL_ID", "irodori-tts-mlx"),
        weights_dir=os.getenv("IRODORI_MLX_WEIGHTS_DIR"),
        weights_repo=os.getenv("IRODORI_MLX_WEIGHTS_REPO"),
        weights_revision=os.getenv("IRODORI_MLX_WEIGHTS_REVISION"),
        text_tokenizer_repo=os.getenv("IRODORI_MLX_TEXT_TOKENIZER_REPO"),
        caption_tokenizer_repo=os.getenv("IRODORI_MLX_CAPTION_TOKENIZER_REPO"),
        text_max_length=_env_int("IRODORI_MLX_TEXT_MAX_LENGTH", default=256) or 256,
        caption_max_length=_env_int("IRODORI_MLX_CAPTION_MAX_LENGTH"),
        codec_repo=os.getenv("IRODORI_MLX_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim"),
        codec_device=os.getenv("IRODORI_MLX_CODEC_DEVICE", "cpu"),
        codec_runtime_mode=os.getenv("IRODORI_MLX_CODEC_RUNTIME_MODE", "persistent"),
        enable_watermark=_env_bool("IRODORI_MLX_ENABLE_WATERMARK"),
        normalize_codec_audio=not _env_bool("IRODORI_MLX_DISABLE_CODEC_NORMALIZE"),
        preload=_env_bool("IRODORI_MLX_PRELOAD"),
    )


def create_default_runtime(config: IrodoriRuntimeConfig | None = None) -> SpeechRuntime:
    try:
        runtime_config = config or runtime_config_from_env()
    except RuntimeUnavailableError as exc:
        return InvalidConfigurationSpeechRuntime(str(exc))
    if not runtime_config.configured:
        return UnconfiguredSpeechRuntime()
    try:
        return IrodoriMLXRuntimeManager(runtime_config)
    except RuntimeUnavailableError as exc:
        return InvalidConfigurationSpeechRuntime(str(exc), model_id=runtime_config.model_id)


def _import_runtime_modules() -> tuple[Any, Callable[..., Any]]:
    try:
        runtime_module = importlib.import_module("irodori_mlx.runtime")
        hosted_weights_module = importlib.import_module("irodori_mlx.hosted_weights")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "Irodori-TTS-MLX runtime dependencies are not installed. Install the "
            "Irodori-TTS-MLX package with its runtime extras in this environment."
        ) from exc
    return runtime_module, hosted_weights_module.resolve_weights_layout_source


def _clean_option(options: dict[str, Any], key: str, *, default: Any = None) -> Any:
    value = options.get(key, default)
    return default if value == "" else value


def _string_option(options: dict[str, Any], key: str, *, default: str | None = None) -> str | None:
    value = _clean_option(options, key, default=default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeRequestError(f"irodori.{key} must be a string.")
    return value


def _bool_option(options: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = options.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise RuntimeRequestError(f"irodori.{key} must be a boolean.")


def _int_option(options: dict[str, Any], key: str, *, default: int, minimum: int = 1) -> int:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise RuntimeRequestError(f"irodori.{key} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeRequestError(f"irodori.{key} must be an integer.") from exc
    if parsed < minimum:
        raise RuntimeRequestError(f"irodori.{key} must be >= {minimum}.")
    return parsed


def _float_option(
    options: dict[str, Any], key: str, *, default: float | None = None, positive: bool = True
) -> float | None:
    value = options.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeRequestError(f"irodori.{key} must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeRequestError(f"irodori.{key} must be a number.") from exc
    if positive and parsed <= 0:
        raise RuntimeRequestError(f"irodori.{key} must be > 0.")
    return parsed


def _required_float_option(
    options: dict[str, Any], key: str, *, default: float, positive: bool = True
) -> float:
    value = _float_option(options, key, default=default, positive=positive)
    assert value is not None
    return value


def _reject_unsupported_lora_adapter(options: dict[str, Any]) -> None:
    adapter = _clean_option(options, LORA_ADAPTER_OPTION)
    if adapter is None:
        return
    if not isinstance(adapter, str):
        raise RuntimeRequestError("irodori.lora_adapter must be a string.")
    adapter = adapter.strip()
    if not adapter:
        return
    if _looks_like_local_path(adapter):
        raise RuntimeRequestError(
            "irodori.lora_adapter must be a configured adapter alias; arbitrary local paths are not accepted."
        )
    raise RuntimeRequestError(
        "irodori.lora_adapter is not supported by the current Irodori-TTS-MLX runtime boundary."
    )


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "~")) or "/" in value or "\\" in value or ":" in value


def _num_steps_option(options: dict[str, Any]) -> int:
    preset = _string_option(options, "preset")
    if preset is not None and preset not in PRESET_NUM_STEPS:
        choices = "', '".join(PRESET_NUM_STEPS)
        raise RuntimeRequestError(f"irodori.preset must be one of '{choices}'.")
    default = PRESET_NUM_STEPS[preset] if preset else 40
    return _int_option(options, "num_steps", default=default)


def split_text_for_generation(text: str, *, max_chars: int) -> list[str]:
    """Split long text on natural boundaries before falling back to hard slices."""

    if max_chars < 1:
        raise RuntimeRequestError("irodori.chunk_max_chars must be >= 1.")
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for segment in _iter_punctuation_segments(text):
        if not current:
            current = segment
            continue
        if len(current) + len(segment) <= max_chars:
            current += segment
            continue
        chunks.extend(_split_overlong_segment(current, max_chars=max_chars))
        current = segment.lstrip()
    if current:
        chunks.extend(_split_overlong_segment(current, max_chars=max_chars))
    return [chunk for chunk in chunks if chunk]


def _iter_punctuation_segments(text: str) -> list[str]:
    break_chars = set("。．.!！？?、，,;；:\n")
    segments: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in break_chars:
            end = index + 1
            segment = text[start:end]
            if segment:
                segments.append(segment)
            start = end
    tail = text[start:]
    if tail:
        segments.append(tail)
    return segments


def _split_overlong_segment(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if text.strip() == "":
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


@dataclass(frozen=True)
class TailArtifactOptions:
    trim_ms: int = DEFAULT_TAIL_TRIM_MS
    silence_trim_ms: int = DEFAULT_TAIL_SILENCE_TRIM_MS
    silence_keep_ms: int = DEFAULT_TAIL_SILENCE_KEEP_MS
    silence_threshold: int = DEFAULT_TAIL_SILENCE_THRESHOLD


def parse_tail_artifact_options(options: dict[str, Any]) -> TailArtifactOptions:
    trim_ms = _int_option(options, "tail_trim_ms", default=DEFAULT_TAIL_TRIM_MS, minimum=0)
    silence_trim_ms = _int_option(
        options, "tail_silence_trim_ms", default=DEFAULT_TAIL_SILENCE_TRIM_MS, minimum=0
    )
    silence_keep_ms = _int_option(
        options, "tail_silence_keep_ms", default=DEFAULT_TAIL_SILENCE_KEEP_MS, minimum=0
    )
    silence_threshold = _int_option(
        options,
        "tail_silence_threshold",
        default=DEFAULT_TAIL_SILENCE_THRESHOLD,
        minimum=0,
    )
    return TailArtifactOptions(
        trim_ms=trim_ms,
        silence_trim_ms=silence_trim_ms,
        silence_keep_ms=silence_keep_ms,
        silence_threshold=silence_threshold,
    )


def process_wav_tail(audio: bytes, options: TailArtifactOptions) -> bytes:
    if options.trim_ms == 0 and options.silence_trim_ms == 0:
        return audio
    params, frames = _read_wav_frames(audio)
    frame_width = params.sampwidth * params.nchannels
    trim_frames = min(params.nframes, _ms_to_frames(options.trim_ms, framerate=params.framerate))
    if trim_frames:
        frames = frames[: -(trim_frames * frame_width)]
    if options.silence_trim_ms:
        frames = _trim_trailing_silence(
            frames,
            params=params,
            minimum_silence_frames=_ms_to_frames(
                options.silence_trim_ms, framerate=params.framerate
            ),
            keep_silence_frames=_ms_to_frames(options.silence_keep_ms, framerate=params.framerate),
            threshold=options.silence_threshold,
        )
    return _write_wav_frames(params, frames)


def concatenate_wav_audio(parts: Sequence[bytes]) -> bytes:
    if not parts:
        raise RuntimeUnavailableError("Irodori-TTS-MLX generation produced no audio chunks.")
    if len(parts) == 1:
        return parts[0]
    first_params, first_frames = _read_wav_frames(parts[0])
    frames = [first_frames]
    for part in parts[1:]:
        params, chunk_frames = _read_wav_frames(part)
        if params[:3] != first_params[:3] or params.framerate != first_params.framerate:
            raise RuntimeUnavailableError(
                "Generated WAV chunks have incompatible audio parameters."
            )
        frames.append(chunk_frames)
    return _write_wav_frames(first_params, b"".join(frames))


def _read_wav_frames(audio: bytes) -> tuple[wave._wave_params, bytes]:
    try:
        with wave.open(BytesIO(audio), "rb") as wav_file:
            params = wav_file.getparams()
            return params, wav_file.readframes(params.nframes)
    except (EOFError, wave.Error) as exc:
        raise RuntimeUnavailableError(
            f"Irodori-TTS-MLX generated invalid WAV audio: {exc}"
        ) from exc


def _write_wav_frames(params: wave._wave_params, frames: bytes) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(frames)
    return output.getvalue()


def _ms_to_frames(milliseconds: int, *, framerate: int) -> int:
    return round(framerate * milliseconds / 1000)


def _trim_trailing_silence(
    frames: bytes,
    *,
    params: wave._wave_params,
    minimum_silence_frames: int,
    keep_silence_frames: int,
    threshold: int,
) -> bytes:
    if params.sampwidth != 2 or minimum_silence_frames <= 0:
        return frames
    frame_width = params.sampwidth * params.nchannels
    trailing_silent_frames = 0
    for frame_start in range(len(frames) - frame_width, -1, -frame_width):
        frame = frames[frame_start : frame_start + frame_width]
        samples = [
            int.from_bytes(frame[index : index + params.sampwidth], "little", signed=True)
            for index in range(0, len(frame), params.sampwidth)
        ]
        if all(abs(sample) <= threshold for sample in samples):
            trailing_silent_frames += 1
            continue
        break
    if trailing_silent_frames < minimum_silence_frames:
        return frames
    remove_frames = max(0, trailing_silent_frames - keep_silence_frames)
    return frames[: -(remove_frames * frame_width)] if remove_frames else frames


class IrodoriMLXRuntimeManager:
    """Lazy, cacheable adapter from server requests to Irodori-TTS-MLX generation."""

    def __init__(
        self,
        config: IrodoriRuntimeConfig,
        *,
        runtime_factory: Callable[..., Any] | None = None,
        module_loader: Callable[[], tuple[Any, Callable[..., Any]]] = _import_runtime_modules,
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory
        self._module_loader = module_loader
        self._runtime: Any | None = None
        self._runtime_lock = threading.Lock()
        self._load_error: str | None = None
        self._loading = False
        if config.preload:
            self._get_runtime()

    def list_models(self) -> list[str]:
        return [self.config.model_id]

    def status_metadata(self) -> dict[str, Any]:
        source = "weights_dir"
        if self.config.weights_repo:
            source = "weights_repo"
        return {
            "runtime": "irodori-tts-mlx",
            "configured": self.config.configured,
            "loaded": self._runtime is not None,
            "load_state": self._load_state(),
            "model_id": self.config.model_id,
            "weights_source": source,
            "last_load_error": self._load_error,
        }

    def _load_state(self) -> str:
        if self._runtime is not None:
            return "loaded"
        if self._loading:
            return "loading"
        if self._load_error is not None:
            return "failed"
        return "not_loaded"

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        _reject_unsupported_lora_adapter(dict(request.irodori))
        runtime = self._get_runtime()
        options = dict(request.irodori)
        chunks = self._split_request_text(request, options)
        chunk_seconds = self._chunk_seconds(options, chunks)
        tail_options = parse_tail_artifact_options(options)
        audio_chunks: list[bytes] = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_irodori = dict(request.irodori)
            chunk_irodori["seconds"] = chunk_seconds[chunk_index]
            chunk_request = SpeechGenerationRequest(
                model=request.model,
                input=chunk,
                voice=request.voice,
                response_format=request.response_format,
                speed=request.speed,
                irodori=chunk_irodori,
            )
            audio_chunks.append(
                process_wav_tail(self._generate_single_chunk(runtime, chunk_request), tail_options)
            )
        return SpeechGenerationResult(
            audio=concatenate_wav_audio(audio_chunks), media_type=WAV_MEDIA_TYPE
        )

    def _generate_single_chunk(self, runtime: Any, request: SpeechGenerationRequest) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = Path(output_file.name)
        try:
            try:
                generation_request = self._build_generation_request(request, output_path)
            except ValueError as exc:
                raise RuntimeRequestError(str(exc)) from exc
            runtime.generate(generation_request)
            return output_path.read_bytes()
        except RuntimeRequestError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeUnavailableError(f"Irodori-TTS-MLX generation failed: {exc}") from exc
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _split_request_text(
        self, request: SpeechGenerationRequest, options: dict[str, Any]
    ) -> list[str]:
        chunking = _bool_option(options, "chunking", default=True)
        if not chunking:
            return [request.input]
        chunk_max_chars = _int_option(
            options, "chunk_max_chars", default=self.config.text_max_length, minimum=1
        )
        return split_text_for_generation(request.input, max_chars=chunk_max_chars)

    def _chunk_seconds(self, options: dict[str, Any], chunks: list[str]) -> list[float | None]:
        seconds = _float_option(options, "seconds", default=None)
        if seconds is None:
            return [None] * len(chunks)
        total_chars = sum(len(chunk) for chunk in chunks)
        if len(chunks) == 1 or total_chars == 0:
            return [seconds]
        remaining = seconds
        chunk_seconds: list[float] = []
        for chunk in chunks[:-1]:
            value = seconds * len(chunk) / total_chars
            chunk_seconds.append(value)
            remaining -= value
        chunk_seconds.append(remaining)
        return chunk_seconds

    def _get_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            try:
                self._loading = True
                logger.info(
                    "runtime_load_start model_id=%s preload=%s configured=%s",
                    self.config.model_id,
                    self.config.preload,
                    self.config.configured,
                )
                self._runtime = self._build_runtime()
            except RuntimeUnavailableError as exc:
                self._load_error = str(exc)
                logger.error("runtime_load_failed model_id=%s error=%s", self.config.model_id, exc)
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                message = f"Irodori-TTS-MLX runtime could not be loaded: {exc}"
                self._load_error = message
                logger.error("runtime_load_failed model_id=%s error=%s", self.config.model_id, exc)
                raise RuntimeUnavailableError(message) from exc
            finally:
                self._loading = False
            self._load_error = None
            logger.info("runtime_load_complete model_id=%s", self.config.model_id)
            return self._runtime

    def _build_runtime(self) -> Any:
        runtime_module, resolve_weights_layout_source = self._module_loader()
        layout = resolve_weights_layout_source(
            weights_dir=self.config.weights_dir,
            weights_repo=self.config.weights_repo,
            revision=self.config.weights_revision,
        )
        if layout is None:
            raise RuntimeUnavailableError(
                "No hosted MLX weights layout is configured. Set IRODORI_MLX_WEIGHTS_DIR "
                "or IRODORI_MLX_WEIGHTS_REPO."
            )
        weights_path = str(layout.weights_path)
        model_config = layout.model_config

        runtime_config = runtime_module.MLXRuntimeConfig(
            model_config=model_config,
            weights_path=weights_path,
            text_tokenizer_repo=self.config.text_tokenizer_repo,
            caption_tokenizer_repo=self.config.caption_tokenizer_repo,
            text_max_length=self.config.text_max_length,
            caption_max_length=self.config.caption_max_length,
            codec=runtime_module.DACVAEBridgeConfig(
                codec_repo=self.config.codec_repo,
                codec_device=self.config.codec_device,
                runtime_mode=self.config.codec_runtime_mode,
                enable_watermark=self.config.enable_watermark,
                normalize_db=-16.0 if self.config.normalize_codec_audio else None,
            ),
        )
        factory = self._runtime_factory or runtime_module.MLXDACVAERuntime
        return factory(config=runtime_config)

    def _build_generation_request(self, request: SpeechGenerationRequest, output_path: Path) -> Any:
        runtime_module, _ = self._module_loader()
        options = dict(request.irodori)
        if request.response_format != "wav":
            raise RuntimeRequestError(
                "Only wav output can be passed to the Irodori-TTS-MLX runtime."
            )
        _reject_unsupported_lora_adapter(options)
        reference_wav = _string_option(options, "reference_wav")
        no_reference = _bool_option(options, "no_reference", default=reference_wav is None)
        if reference_wav and no_reference:
            raise RuntimeRequestError(
                "irodori.reference_wav and irodori.no_reference=true cannot both be set."
            )
        if not reference_wav and not no_reference:
            raise RuntimeRequestError("irodori.no_reference=false requires irodori.reference_wav.")
        duration_scale = _float_option(options, "duration_scale", default=None)
        if duration_scale is None:
            duration_scale = 1.0 / request.speed
        cfg_min_t = _required_float_option(options, "cfg_min_t", default=0.5, positive=False)
        cfg_max_t = _required_float_option(options, "cfg_max_t", default=1.0, positive=False)
        if cfg_min_t > cfg_max_t:
            raise RuntimeRequestError("irodori.cfg_min_t must be <= irodori.cfg_max_t.")
        return runtime_module.GenerationRequest(
            text=request.input,
            output_wav=str(output_path),
            reference_wav=reference_wav,
            no_reference=no_reference,
            caption=_string_option(options, "caption"),
            seconds=_float_option(options, "seconds", default=None),
            duration_scale=duration_scale,
            num_steps=_num_steps_option(options),
            cfg_scale_text=_required_float_option(options, "cfg_scale_text", default=3.0),
            cfg_scale_caption=_required_float_option(options, "cfg_scale_caption", default=3.0),
            cfg_scale_speaker=_required_float_option(options, "cfg_scale_speaker", default=5.0),
            cfg_guidance_mode=_string_option(options, "cfg_guidance_mode", default="independent")
            or "independent",
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            seed=_int_option(options, "seed", default=0, minimum=0),
            max_reference_seconds=_float_option(options, "max_reference_seconds", default=30.0),
            use_context_kv_cache=not _bool_option(options, "no_context_kv_cache", default=False),
        )

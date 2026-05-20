"""Runtime boundary for speech generation."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import tempfile
import threading
import wave
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

PRESET_NUM_STEPS = {
    "ultra-fast": 8,
    "fast": 12,
    "balanced": 24,
    "quality": 40,
}
ULTRA_FAST_SHORT_PROMPT_MAX_AUTO_SECONDS = 2.5
ULTRA_FAST_SHORT_PROMPT_MAX_ESTIMATE_SECONDS = 3.0
DEFAULT_TAIL_TRIM_MS = 0
DEFAULT_TAIL_SILENCE_TRIM_MS = 0
DEFAULT_TAIL_SILENCE_KEEP_MS = 40
DEFAULT_TAIL_SILENCE_THRESHOLD = 256
WAV_MEDIA_TYPE = "audio/wav"
LORA_ADAPTER_OPTION = "lora_adapter"
DEFAULT_CODEC_ARTIFACT_REPO = "t0yohei/Irodori-TTS-MLX-DACVAE-Codec"
MLX_CODEC_RUNTIME_MODES = {"mlx", "mlx-decode", "mlx-decode-subprocess"}
MANAGED_REFERENCE_CACHE_OPTION = "_managed_reference_cache"
DEFAULT_PUNCTUATION_CHUNK_MIN_CHARS = 80
DEFAULT_PUNCTUATION_CHUNK_TARGET_CHARS = 40
DEFAULT_PUNCTUATION_CHUNK_HARD_MAX_CHARS = 120
FIRST_SENTENCE_COMMA_BREAK_CHARS = set("、，,;；:：")
PUNCTUATION_BREAK_CHARS = set("。．.!！？?、，,;；:\n")
SENTENCE_TERMINAL_CHARS = set("。．.!！？?")
PUNCTUATION_CLOSING_CHARS = set("」』）)]】〕〉》｝}＞>？”’”\"'!?！？")
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


@dataclass(frozen=True)
class ManagedReferenceCacheInfo:
    voice_id: str
    path: str
    size: int
    mtime_ns: int


class ManagedReferenceCache:
    """Bounded in-memory cache for server-managed reference audio latents."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def status_metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.max_entries > 0,
                "max_entries": self.max_entries,
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def clear_voice(self, voice_id: str) -> None:
        with self._lock:
            for key in list(self._entries):
                if key[0] == voice_id:
                    del self._entries[key]

    def get_or_encode(
        self,
        info: ManagedReferenceCacheInfo,
        *,
        max_seconds: float | None,
        normalize_db: float | None,
        ensure_max: bool,
        encode: Callable[[], Any],
    ) -> Any:
        if self.max_entries <= 0:
            return encode()
        key = (
            info.voice_id,
            info.path,
            info.size,
            info.mtime_ns,
            max_seconds,
            normalize_db,
            ensure_max,
        )
        with self._lock:
            if key in self._entries:
                self.hits += 1
                value = self._entries.pop(key)
                self._entries[key] = value
                logger.debug(
                    "managed_reference_cache_hit voice_id=%s path=%s entries=%s",
                    info.voice_id,
                    info.path,
                    len(self._entries),
                )
                return value
            self.misses += 1
        value = encode()
        with self._lock:
            self._entries[key] = value
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1
        logger.debug(
            "managed_reference_cache_miss voice_id=%s path=%s entries=%s",
            info.voice_id,
            info.path,
            len(self._entries),
        )
        return value


class ManagedReferenceCachingBridge:
    """Thread-local cache wrapper around a DACVAE bridge."""

    def __init__(self, bridge: Any, cache: ManagedReferenceCache) -> None:
        self._bridge = bridge
        self._cache = cache
        self._local = threading.local()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bridge, name)

    def configure_cache(self, cache: ManagedReferenceCache) -> None:
        self._cache = cache

    @contextmanager
    def reference_cache(self, info: ManagedReferenceCacheInfo | None) -> Iterator[None]:
        previous = getattr(self._local, "reference_info", None)
        self._local.reference_info = info
        try:
            yield
        finally:
            self._local.reference_info = previous

    def encode_reference(
        self,
        path: str | Path,
        *,
        max_seconds: float | None,
        normalize_db: float | None,
        ensure_max: bool,
    ) -> Any:
        info = getattr(self._local, "reference_info", None)
        path_text = str(path)
        if info is None or path_text != info.path:
            return self._bridge.encode_reference(
                path,
                max_seconds=max_seconds,
                normalize_db=normalize_db,
                ensure_max=ensure_max,
            )
        return self._cache.get_or_encode(
            info,
            max_seconds=max_seconds,
            normalize_db=normalize_db,
            ensure_max=ensure_max,
            encode=lambda: self._bridge.encode_reference(
                path,
                max_seconds=max_seconds,
                normalize_db=normalize_db,
                ensure_max=ensure_max,
            ),
        )


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
    max_text_len: int = 256
    max_caption_len: int | None = None
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    codec_path: str | None = None
    codec_artifact_repo: str | None = DEFAULT_CODEC_ARTIFACT_REPO
    codec_artifact_revision: str | None = None
    codec_device: str = "cpu"
    codec_runtime_mode: str = "mlx"
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
        max_text_len=_env_int("IRODORI_MLX_MAX_TEXT_LEN", default=256) or 256,
        max_caption_len=_env_int("IRODORI_MLX_MAX_CAPTION_LEN"),
        codec_repo=os.getenv("IRODORI_MLX_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim"),
        codec_path=os.getenv("IRODORI_MLX_CODEC_PATH") or None,
        codec_artifact_repo=os.getenv(
            "IRODORI_MLX_CODEC_ARTIFACT_REPO", DEFAULT_CODEC_ARTIFACT_REPO
        )
        or None,
        codec_artifact_revision=os.getenv("IRODORI_MLX_CODEC_ARTIFACT_REVISION") or None,
        codec_device=os.getenv("IRODORI_MLX_CODEC_DEVICE", "cpu"),
        codec_runtime_mode=os.getenv("IRODORI_MLX_CODEC_RUNTIME_MODE", "mlx"),
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


def _import_runtime_modules() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    try:
        runtime_module = importlib.import_module("irodori_mlx.runtime")
        hosted_weights_module = importlib.import_module("irodori_mlx.hosted_weights")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "Irodori-TTS-MLX runtime dependencies are not installed. Install the "
            "Irodori-TTS-MLX package with its runtime extras in this environment."
        ) from exc

    def resolve_codec_artifact_source(**kwargs: Any) -> Any:
        try:
            hosted_codec_module = importlib.import_module("irodori_mlx.hosted_codec")
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Installed Irodori-TTS-MLX runtime does not support hosted codec artifacts. "
                "Upgrade Irodori-TTS-MLX before using MLX DACVAE codec modes."
            ) from exc
        return hosted_codec_module.resolve_codec_artifact_source(**kwargs)

    return (
        runtime_module,
        hosted_weights_module.resolve_weights_layout_source,
        resolve_codec_artifact_source,
    )


def _codec_bridge_config(
    runtime_module: Any,
    config: IrodoriRuntimeConfig,
    *,
    codec_path: str | None,
) -> Any:
    if config.codec_runtime_mode not in MLX_CODEC_RUNTIME_MODES:
        supported = ", ".join(sorted(MLX_CODEC_RUNTIME_MODES))
        raise RuntimeUnavailableError(
            f"Unsupported MLX codec runtime mode '{config.codec_runtime_mode}'. "
            f"Supported modes: {supported}."
        )
    codec_config_class = runtime_module.DACVAEBridgeConfig
    parameters = inspect.signature(codec_config_class).parameters
    if "codec_path" not in parameters:
        raise RuntimeUnavailableError(
            "Installed Irodori-TTS-MLX runtime does not support MLX DACVAE codec "
            "artifacts. Upgrade Irodori-TTS-MLX before using this server."
        )
    codec_kwargs: dict[str, Any] = {
        "codec_repo": config.codec_repo,
        "codec_path": codec_path,
        "codec_device": config.codec_device,
        "runtime_mode": config.codec_runtime_mode,
        "enable_watermark": config.enable_watermark,
        "normalize_db": -16.0 if config.normalize_codec_audio else None,
    }
    return codec_config_class(**codec_kwargs)


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


def _choice_option(
    options: dict[str, Any], key: str, *, default: str, choices: set[str]
) -> str:
    value = _string_option(options, key, default=default) or default
    if value not in choices:
        joined = "', '".join(sorted(choices))
        raise RuntimeRequestError(f"irodori.{key} must be one of '{joined}'.")
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


def _chunk_mode_option(options: dict[str, Any]) -> str:
    return "punctuation"


def _runtime_sampling_request_class(runtime_module: Any) -> Any:
    return _required_runtime_symbol(runtime_module, "SamplingRequest")


def _runtime_class(runtime_module: Any) -> Any:
    return _required_runtime_symbol(runtime_module, "InferenceRuntime")


def _required_runtime_symbol(runtime_module: Any, name: str) -> Any:
    try:
        return getattr(runtime_module, name)
    except AttributeError as exc:
        raise RuntimeUnavailableError(
            "Installed Irodori-TTS-MLX runtime is too old for this server. "
            f"Upgrade Irodori-TTS-MLX so irodori_mlx.runtime exposes {name}."
        ) from exc


def _runtime_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _runtime_generation_request_kwargs(runtime_module: Any, **kwargs: Any) -> dict[str, Any]:
    request_class = _runtime_sampling_request_class(runtime_module)
    signature = inspect.signature(request_class)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _ensure_sampling_request_fields(
    runtime_module: Any, *, requested_fields: dict[str, str]
) -> None:
    request_class = _runtime_sampling_request_class(runtime_module)
    parameters = inspect.signature(request_class).parameters
    missing = [field for field in requested_fields if field not in parameters]
    if not missing:
        return
    options = ", ".join(f"irodori.{requested_fields[field]}" for field in missing)
    fields = ", ".join(missing)
    raise RuntimeUnavailableError(
        "Installed Irodori-TTS-MLX runtime is too old for requested option(s) "
        f"{options}. Upgrade Irodori-TTS-MLX so SamplingRequest exposes {fields}."
    )


def _managed_reference_cache_info(options: dict[str, Any]) -> ManagedReferenceCacheInfo | None:
    value = options.get(MANAGED_REFERENCE_CACHE_OPTION)
    if not isinstance(value, dict):
        return None
    try:
        voice_id = value["voice_id"]
        path = value["path"]
        size = int(value["size"])
        mtime_ns = int(value["mtime_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(voice_id, str) or not isinstance(path, str):
        return None
    return ManagedReferenceCacheInfo(
        voice_id=voice_id,
        path=path,
        size=size,
        mtime_ns=mtime_ns,
    )


def split_text_for_generation(
    text: str,
    *,
    max_chars: int,
    chunk_mode: str = "max_chars",
    chunk_min_chars: int = DEFAULT_PUNCTUATION_CHUNK_MIN_CHARS,
    chunk_target_chars: int = DEFAULT_PUNCTUATION_CHUNK_TARGET_CHARS,
    chunk_hard_max_chars: int = DEFAULT_PUNCTUATION_CHUNK_HARD_MAX_CHARS,
    first_sentence_comma_chunking_enabled: bool = False,
) -> list[str]:
    """Split long text on natural boundaries before falling back to hard slices."""

    if chunk_mode not in {"max_chars", "punctuation"}:
        raise RuntimeRequestError("irodori.chunk_mode must be one of 'max_chars', 'punctuation'.")
    if max_chars < 1:
        raise RuntimeRequestError("max_chars must be >= 1.")
    if chunk_min_chars < 0:
        raise RuntimeRequestError("irodori.chunk_min_chars must be >= 0.")
    if chunk_target_chars < 1:
        raise RuntimeRequestError("irodori.chunk_target_chars must be >= 1.")
    if chunk_hard_max_chars < 1:
        raise RuntimeRequestError("irodori.chunk_hard_max_chars must be >= 1.")
    return _split_text_by_punctuation_plan(
        text,
        min_chars=chunk_min_chars,
        hard_max_chars=min(chunk_hard_max_chars, max_chars),
        split_first_sentence_on_commas=first_sentence_comma_chunking_enabled,
    )


def _split_text_by_punctuation_plan(
    text: str,
    *,
    min_chars: int,
    hard_max_chars: int,
    split_first_sentence_on_commas: bool = False,
) -> list[str]:
    if split_first_sentence_on_commas:
        fast_chunks = _split_first_sentence_on_commas(
            text,
            min_chars=min_chars,
            hard_max_chars=hard_max_chars,
        )
        if fast_chunks is not None:
            return fast_chunks

    if not any(char in text for char in PUNCTUATION_BREAK_CHARS):
        return _split_overlong_segment(text, max_chars=hard_max_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    pending_boundary = False

    for char in text:
        if pending_boundary and char not in PUNCTUATION_CLOSING_CHARS:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
            current_chars = 0
            pending_boundary = False
            if char.isspace():
                continue

        current.append(char)
        if not char.isspace():
            current_chars += 1
        if len(current) >= hard_max_chars:
            chunk = "".join(current)
            chunks.append(chunk if chunk.strip() == "" else chunk.strip())
            current = []
            current_chars = 0
            pending_boundary = False
            continue
        if char in PUNCTUATION_BREAK_CHARS and current_chars >= min_chars:
            pending_boundary = True

    tail = "".join(current).strip()
    if tail:
        chunks.extend(_split_overlong_segment(tail, max_chars=hard_max_chars))
    return [chunk for chunk in chunks if chunk]


def _split_first_sentence_on_commas(
    text: str,
    *,
    min_chars: int,
    hard_max_chars: int,
) -> list[str] | None:
    sentence_end = _first_sentence_end_index(text)
    first_sentence = text[:sentence_end]
    if not any(char in FIRST_SENTENCE_COMMA_BREAK_CHARS for char in first_sentence):
        return None

    chunks: list[str] = []
    start = 0
    for index, char in enumerate(first_sentence):
        if char not in FIRST_SENTENCE_COMMA_BREAK_CHARS:
            continue
        segment = first_sentence[start : index + 1]
        if segment:
            chunks.extend(
                _split_overlong_segment(
                    _lstrip_chunk_segment(segment, chunks),
                    max_chars=hard_max_chars,
                )
            )
        start = index + 1

    tail_segment = first_sentence[start:]
    if tail_segment:
        chunks.extend(
            _split_overlong_segment(
                _lstrip_chunk_segment(tail_segment, chunks),
                max_chars=hard_max_chars,
            )
        )

    remaining = _lstrip_non_whitespace(text[sentence_end:])
    if remaining:
        chunks.extend(
            _split_text_by_punctuation_plan(
                remaining,
                min_chars=min_chars,
                hard_max_chars=hard_max_chars,
                split_first_sentence_on_commas=False,
            )
        )
    return [chunk for chunk in chunks if chunk]


def _lstrip_chunk_segment(segment: str, chunks: Sequence[str]) -> str:
    return _lstrip_non_whitespace(segment) if chunks else segment


def _first_sentence_end_index(text: str) -> int:
    for index, char in enumerate(text):
        if char not in SENTENCE_TERMINAL_CHARS:
            continue
        end = index + 1
        while end < len(text) and text[end] in PUNCTUATION_CLOSING_CHARS:
            end += 1
        return end
    return len(text)


def _lstrip_non_whitespace(text: str) -> str:
    return text if text.strip() == "" else text.lstrip()


def _split_leading_closing_punctuation(text: str) -> tuple[str, str]:
    index = 0
    while index < len(text) and text[index] in PUNCTUATION_CLOSING_CHARS:
        index += 1
    return text[:index], text[index:]


def _ends_with_japanese_period_before_closers(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in PUNCTUATION_CLOSING_CHARS:
        stripped = stripped[:-1]
    return stripped.endswith("。")


def _iter_punctuation_segments(text: str) -> list[str]:
    segments: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in PUNCTUATION_BREAK_CHARS:
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
        module_loader: Callable[[], tuple[Any, Callable[..., Any], Callable[..., Any]]] = (
            _import_runtime_modules
        ),
    ) -> None:
        self.config = config
        self._runtime_factory = runtime_factory
        self._module_loader = module_loader
        self._runtime: Any | None = None
        self._runtime_lock = threading.Lock()
        self._load_error: str | None = None
        self._resolved_codec_source: str | None = None
        self._resolved_codec_source_kind: str | None = None
        self._loading = False
        self._managed_reference_cache = ManagedReferenceCache(max_entries=8)
        self._managed_reference_bridge: ManagedReferenceCachingBridge | None = None
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
            "codec_path_configured": self.config.codec_path is not None,
            "codec_artifact_repo": self.config.codec_artifact_repo,
            "codec_artifact_revision": self.config.codec_artifact_revision,
            "codec_artifact_source": self._resolved_codec_source,
            "codec_artifact_source_kind": self._resolved_codec_source_kind,
            "codec_runtime_mode": self.config.codec_runtime_mode,
            "last_load_error": self._load_error,
            "managed_reference_cache": self._managed_reference_cache.status_metadata(),
        }

    def configure_managed_reference_cache(self, *, max_entries: int) -> None:
        self._managed_reference_cache = ManagedReferenceCache(max_entries=max_entries)
        self._managed_reference_bridge = None
        if self._runtime is not None:
            self._install_managed_reference_cache_bridge(self._runtime)

    def invalidate_managed_reference_cache(self, voice_id: str) -> None:
        self._managed_reference_cache.clear_voice(voice_id)

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
            if "seconds" in chunk_irodori or chunk_seconds[chunk_index] is not None:
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
            except RuntimeUnavailableError:
                raise
            cache_info = _managed_reference_cache_info(request.irodori)
            cache_context = (
                self._managed_reference_bridge.reference_cache(cache_info)
                if self._managed_reference_bridge is not None
                else nullcontext()
            )
            with cache_context:
                runtime.generate(generation_request)
            return output_path.read_bytes()
        except RuntimeRequestError:
            raise
        except RuntimeUnavailableError:
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
        chunking = _bool_option(
            options,
            "chunking_enabled",
            default=True,
        )
        if not chunking:
            return [request.input]
        chunk_mode = _chunk_mode_option(options)
        chunk_min_chars = _int_option(
            options,
            "chunk_min_chars",
            default=DEFAULT_PUNCTUATION_CHUNK_MIN_CHARS,
            minimum=0,
        )
        return split_text_for_generation(
            request.input,
            max_chars=self.config.max_text_len,
            chunk_mode=chunk_mode,
            chunk_min_chars=chunk_min_chars,
            first_sentence_comma_chunking_enabled=_bool_option(
                options,
                "first_sentence_comma_chunking_enabled",
            ),
        )

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
                self._install_managed_reference_cache_bridge(self._runtime)
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
        runtime_module, resolve_weights_layout_source, resolve_codec_artifact_source = (
            self._module_loader()
        )
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
        codec_path = self._resolve_codec_path(resolve_codec_artifact_source)

        runtime_config = runtime_module.MLXRuntimeConfig(
            **_runtime_kwargs(
                runtime_module.MLXRuntimeConfig,
                model_config=model_config,
                weights_path=weights_path,
                text_tokenizer_repo=self.config.text_tokenizer_repo,
                caption_tokenizer_repo=self.config.caption_tokenizer_repo,
                max_text_len=self.config.max_text_len,
                max_caption_len=self.config.max_caption_len,
                codec=_codec_bridge_config(runtime_module, self.config, codec_path=codec_path),
            )
        )
        factory = self._runtime_factory or _runtime_class(runtime_module)
        return factory(config=runtime_config)

    def _install_managed_reference_cache_bridge(self, runtime: Any) -> None:
        bridge = getattr(runtime, "bridge", None)
        if bridge is None:
            return
        if isinstance(bridge, ManagedReferenceCachingBridge):
            bridge.configure_cache(self._managed_reference_cache)
            self._managed_reference_bridge = bridge
            return
        proxy = ManagedReferenceCachingBridge(bridge, self._managed_reference_cache)
        runtime.bridge = proxy
        self._managed_reference_bridge = proxy

    def _resolve_codec_path(self, resolve_codec_artifact_source: Callable[..., Any]) -> str | None:
        if self.config.codec_path:
            self._resolved_codec_source = self.config.codec_path
            self._resolved_codec_source_kind = "path"
            return self.config.codec_path
        if not self.config.codec_artifact_repo:
            raise RuntimeUnavailableError(
                "Irodori-TTS-MLX requires an MLX DACVAE codec artifact. Set "
                "IRODORI_MLX_CODEC_PATH or IRODORI_MLX_CODEC_ARTIFACT_REPO."
            )
        try:
            layout = resolve_codec_artifact_source(
                codec_artifact_repo=self.config.codec_artifact_repo,
                revision=self.config.codec_artifact_revision,
            )
        except ValueError as exc:
            raise RuntimeUnavailableError(
                f"Irodori-TTS-MLX codec artifact could not be loaded: {exc}"
            ) from exc
        if layout is None:
            raise RuntimeUnavailableError(
                "Irodori-TTS-MLX codec artifact resolver returned no layout. Set "
                "IRODORI_MLX_CODEC_PATH or IRODORI_MLX_CODEC_ARTIFACT_REPO."
            )
        self._resolved_codec_source = getattr(layout, "source", self.config.codec_artifact_repo)
        self._resolved_codec_source_kind = getattr(layout, "source_kind", "repo")
        return str(layout.codec_path)

    def _build_generation_request(self, request: SpeechGenerationRequest, output_path: Path) -> Any:
        runtime_module, _, _ = self._module_loader()
        options = dict(request.irodori)
        if request.response_format != "wav":
            raise RuntimeRequestError(
                "Only wav output can be passed to the Irodori-TTS-MLX runtime."
            )
        _reject_unsupported_lora_adapter(options)
        ref_wav = _string_option(options, "ref_wav")
        ref_embed = _string_option(options, "ref_embed")
        no_ref = _bool_option(options, "no_ref", default=ref_wav is None and ref_embed is None)
        selected_reference_inputs = [
            name
            for name, value in (
                ("ref_wav", ref_wav),
                ("ref_embed", ref_embed),
                ("no_ref", no_ref),
            )
            if bool(value)
        ]
        if len(selected_reference_inputs) > 1:
            raise RuntimeRequestError(
                "irodori reference options cannot both be set. Choose only one of "
                "irodori.ref_wav, irodori.ref_embed, or irodori.no_ref=true."
            )
        if not selected_reference_inputs:
            raise RuntimeRequestError("irodori.no_ref=false requires irodori.ref_wav or irodori.ref_embed.")
        duration_scale_explicit = "duration_scale" in options
        duration_scale = _float_option(options, "duration_scale", default=None)
        if duration_scale is None:
            duration_scale = 1.0 / request.speed
        cfg_min_t = _required_float_option(options, "cfg_min_t", default=0.5, positive=False)
        cfg_max_t = _required_float_option(options, "cfg_max_t", default=1.0, positive=False)
        if cfg_min_t > cfg_max_t:
            raise RuntimeRequestError("irodori.cfg_min_t must be <= irodori.cfg_max_t.")
        t_schedule_mode = _choice_option(
            options, "t_schedule_mode", default="linear", choices={"linear", "sway"}
        )
        rescale_k = _float_option(options, "rescale_k", default=None)
        rescale_sigma = _float_option(options, "rescale_sigma", default=None)
        if (rescale_k is None) != (rescale_sigma is None):
            raise RuntimeRequestError("irodori.rescale_k and irodori.rescale_sigma must be set together.")
        speaker_kv_scale = _float_option(options, "speaker_kv_scale", default=None)
        speaker_kv_min_t = None
        if speaker_kv_scale is not None:
            speaker_kv_min_t = _float_option(
                options, "speaker_kv_min_t", default=0.9, positive=False
            )
            assert speaker_kv_min_t is not None
            if not 0.0 <= speaker_kv_min_t <= 1.0:
                raise RuntimeRequestError("irodori.speaker_kv_min_t must be in [0, 1].")
        elif "speaker_kv_min_t" in options:
            speaker_kv_min_t = _float_option(options, "speaker_kv_min_t", positive=False)
        speaker_kv_max_layers = None
        if "speaker_kv_max_layers" in options:
            speaker_kv_max_layers = _int_option(
                options, "speaker_kv_max_layers", default=0, minimum=0
            )
        requested_new_fields = {
            field: option
            for field, option in (
                ("ref_embed", "ref_embed"),
                ("t_schedule_mode", "t_schedule_mode"),
                ("sway_coeff", "sway_coeff"),
                ("rescale_k", "rescale_k"),
                ("rescale_sigma", "rescale_sigma"),
                ("speaker_kv_scale", "speaker_kv_scale"),
                ("speaker_kv_min_t", "speaker_kv_min_t"),
                ("speaker_kv_max_layers", "speaker_kv_max_layers"),
            )
            if option in options
        }
        _ensure_sampling_request_fields(runtime_module, requested_fields=requested_new_fields)
        seconds_explicit = "seconds" in options
        seconds = _float_option(options, "seconds", default=None)
        max_auto_seconds = None
        max_auto_estimate_seconds = None
        if (
            _string_option(options, "preset") == "ultra-fast"
            and not seconds_explicit
            and not duration_scale_explicit
            and request.speed == 1.0
        ):
            max_auto_seconds = ULTRA_FAST_SHORT_PROMPT_MAX_AUTO_SECONDS
            max_auto_estimate_seconds = ULTRA_FAST_SHORT_PROMPT_MAX_ESTIMATE_SECONDS
        request_class = _runtime_sampling_request_class(runtime_module)
        return request_class(
            **_runtime_generation_request_kwargs(
                runtime_module,
                text=request.input,
                output_wav=str(output_path),
                ref_wav=ref_wav,
                ref_embed=ref_embed,
                no_ref=no_ref,
                caption=_string_option(options, "caption"),
                seconds=seconds,
                duration_scale=duration_scale,
                max_auto_seconds=max_auto_seconds,
                max_auto_estimate_seconds=max_auto_estimate_seconds,
                num_steps=_num_steps_option(options),
                cfg_scale_text=_required_float_option(options, "cfg_scale_text", default=3.0),
                cfg_scale_caption=_required_float_option(options, "cfg_scale_caption", default=3.0),
                cfg_scale_speaker=_required_float_option(options, "cfg_scale_speaker", default=5.0),
                cfg_guidance_mode=_string_option(
                    options, "cfg_guidance_mode", default="independent"
                )
                or "independent",
                cfg_min_t=cfg_min_t,
                cfg_max_t=cfg_max_t,
                t_schedule_mode=t_schedule_mode,
                sway_coeff=_required_float_option(
                    options, "sway_coeff", default=-1.0, positive=False
                ),
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers,
                seed=_int_option(options, "seed", default=0, minimum=0),
                max_ref_seconds=_float_option(options, "max_ref_seconds", default=30.0),
                context_kv_cache=_bool_option(options, "context_kv_cache", default=True),
            )
        )

"""Runtime boundary for speech generation."""

from __future__ import annotations

import importlib
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

PRESET_NUM_STEPS = {
    "fast": 12,
    "balanced": 24,
    "quality": 40,
}


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
            "model_id": "irodori-tts-mlx",
        }

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeUnavailableError(
            "Irodori-TTS-MLX runtime is not configured. Set IRODORI_MLX_WEIGHTS_PATH "
            "with IRODORI_MLX_MODEL_CONFIG_JSON, or set IRODORI_MLX_WEIGHTS_DIR / "
            "IRODORI_MLX_WEIGHTS_REPO for a hosted converted weights layout."
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
            "model_id": self.model_id,
            "last_load_error": self.message,
        }

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeUnavailableError(self.message)


@dataclass(frozen=True)
class IrodoriRuntimeConfig:
    """Configuration for the real Irodori-TTS-MLX runtime adapter."""

    model_id: str = "irodori-tts-mlx"
    weights_path: str | None = None
    model_config_json: str | None = None
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
        return bool(self.weights_path or self.weights_dir or self.weights_repo)


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
        weights_path=os.getenv("IRODORI_MLX_WEIGHTS_PATH"),
        model_config_json=os.getenv("IRODORI_MLX_MODEL_CONFIG_JSON"),
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


def _num_steps_option(options: dict[str, Any]) -> int:
    preset = _string_option(options, "preset")
    if preset is not None and preset not in PRESET_NUM_STEPS:
        choices = "', '".join(PRESET_NUM_STEPS)
        raise RuntimeRequestError(f"irodori.preset must be one of '{choices}'.")
    default = PRESET_NUM_STEPS[preset] if preset else 40
    return _int_option(options, "num_steps", default=default)


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
        if config.preload:
            self._get_runtime()

    def list_models(self) -> list[str]:
        return [self.config.model_id]

    def status_metadata(self) -> dict[str, Any]:
        source = "weights_path"
        if self.config.weights_repo:
            source = "weights_repo"
        elif self.config.weights_dir:
            source = "weights_dir"
        return {
            "runtime": "irodori-tts-mlx",
            "configured": self.config.configured,
            "loaded": self._runtime is not None,
            "model_id": self.config.model_id,
            "weights_source": source,
            "last_load_error": self._load_error,
        }

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        runtime = self._get_runtime()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = Path(output_file.name)
        try:
            try:
                generation_request = self._build_generation_request(request, output_path)
            except ValueError as exc:
                raise RuntimeRequestError(str(exc)) from exc
            runtime.generate(generation_request)
            return SpeechGenerationResult(audio=output_path.read_bytes(), media_type="audio/wav")
        except RuntimeRequestError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeUnavailableError(f"Irodori-TTS-MLX generation failed: {exc}") from exc
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _get_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            try:
                self._runtime = self._build_runtime()
            except RuntimeUnavailableError as exc:
                self._load_error = str(exc)
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                message = f"Irodori-TTS-MLX runtime could not be loaded: {exc}"
                self._load_error = message
                raise RuntimeUnavailableError(message) from exc
            self._load_error = None
            return self._runtime

    def _build_runtime(self) -> Any:
        runtime_module, resolve_weights_layout_source = self._module_loader()
        layout = resolve_weights_layout_source(
            weights_dir=self.config.weights_dir,
            weights_repo=self.config.weights_repo,
            revision=self.config.weights_revision,
        )
        if layout is not None:
            weights_path = str(layout.weights_path)
            model_config = layout.model_config
        else:
            if not self.config.weights_path:
                raise RuntimeUnavailableError(
                    "No MLX weights are configured. Set IRODORI_MLX_WEIGHTS_PATH, "
                    "IRODORI_MLX_WEIGHTS_DIR, or IRODORI_MLX_WEIGHTS_REPO."
                )
            if not self.config.model_config_json:
                raise RuntimeUnavailableError(
                    "IRODORI_MLX_MODEL_CONFIG_JSON is required when using IRODORI_MLX_WEIGHTS_PATH."
                )
            weights_path = self.config.weights_path
            model_config = runtime_module.load_model_config_json(self.config.model_config_json)

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
            raise RuntimeRequestError("Only wav output can be passed to the Irodori-TTS-MLX runtime.")
        reference_wav = _string_option(options, "reference_wav")
        no_reference = _bool_option(options, "no_reference", default=reference_wav is None)
        if reference_wav and no_reference:
            raise RuntimeRequestError("irodori.reference_wav and irodori.no_reference=true cannot both be set.")
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
            cfg_guidance_mode=_string_option(options, "cfg_guidance_mode", default="independent") or "independent",
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            seed=_int_option(options, "seed", default=0, minimum=0),
            max_reference_seconds=_float_option(options, "max_reference_seconds", default=30.0),
            use_context_kv_cache=not _bool_option(options, "no_context_kv_cache", default=False),
        )

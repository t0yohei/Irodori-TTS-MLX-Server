from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
import threading
import wave

import pytest

from irodori_tts_mlx_server.runtime import (
    IrodoriMLXRuntimeManager,
    IrodoriRuntimeConfig,
    RuntimeRequestError,
    RuntimeUnavailableError,
    SpeechGenerationRequest,
    concatenate_wav_audio,
    split_text_for_generation,
    create_default_runtime,
)


@dataclass(frozen=True)
class FakeGenerationRequest:
    text: str
    output_wav: str
    reference_wav: str | None = None
    no_reference: bool = False
    caption: str | None = None
    seconds: float | None = None
    duration_scale: float = 1.0
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    seed: int = 0
    max_reference_seconds: float | None = 30.0
    use_context_kv_cache: bool = True


@dataclass(frozen=True)
class FakeRuntimeConfig:
    model_config: object
    weights_path: str
    text_tokenizer_repo: str | None
    caption_tokenizer_repo: str | None
    text_max_length: int
    caption_max_length: int | None
    codec: object


@dataclass(frozen=True)
class FakeCodecConfig:
    codec_repo: str
    codec_device: str
    runtime_mode: str
    enable_watermark: bool
    normalize_db: float | None


class FakeMLXRuntime:
    instances: list["FakeMLXRuntime"] = []

    def __init__(self, *, config: FakeRuntimeConfig) -> None:
        self.config = config
        self.requests: list[FakeGenerationRequest] = []
        FakeMLXRuntime.instances.append(self)

    def generate(self, request: FakeGenerationRequest) -> object:
        self.requests.append(request)
        with open(request.output_wav, "wb") as fh:
            fh.write(b"RIFFfakeWAVE")
        return SimpleNamespace(output_wav=request.output_wav)


class WaveMLXRuntime:
    instances: list["WaveMLXRuntime"] = []

    def __init__(self, *, config: FakeRuntimeConfig) -> None:
        self.config = config
        self.requests: list[FakeGenerationRequest] = []
        WaveMLXRuntime.instances.append(self)

    def generate(self, request: FakeGenerationRequest) -> object:
        self.requests.append(request)
        with open(request.output_wav, "wb") as fh:
            fh.write(make_wav([len(request.text)] * 20 + [0] * 10))
        return SimpleNamespace(output_wav=request.output_wav)


class ValueErrorMLXRuntime:
    def __init__(self, *, config: FakeRuntimeConfig) -> None:
        self.config = config

    def generate(self, request: FakeGenerationRequest) -> object:
        raise ValueError("backend failed")


def fake_module_loader():
    runtime_module = SimpleNamespace(
        DACVAEBridgeConfig=FakeCodecConfig,
        GenerationRequest=FakeGenerationRequest,
        MLXDACVAERuntime=FakeMLXRuntime,
        MLXRuntimeConfig=FakeRuntimeConfig,
    )
    layout = SimpleNamespace(
        weights_path="/weights/model.npz",
        model_config={"family": "voicedesign"},
    )
    return runtime_module, lambda **_kwargs: layout


def hosted_config(**kwargs) -> IrodoriRuntimeConfig:
    return IrodoriRuntimeConfig(weights_repo="owner/repo", **kwargs)


def make_wav(samples: list[int], *, framerate: int = 1000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(
            b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        )
    return output.getvalue()


def read_wav_samples(audio: bytes) -> list[int]:
    with wave.open(BytesIO(audio), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return [
        int.from_bytes(frames[index : index + 2], "little", signed=True)
        for index in range(0, len(frames), 2)
    ]


def test_mlx_runtime_manager_maps_request_options_and_caches_runtime() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            text_tokenizer_repo="text-tokenizer",
            caption_tokenizer_repo="caption-tokenizer",
            text_max_length=128,
            caption_max_length=64,
            codec_device="cpu",
        ),
        module_loader=fake_module_loader,
    )

    result = manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=2.0,
            irodori={
                "caption": "warm voice",
                "no_reference": True,
                "seconds": 1.5,
                "num_steps": 12,
                "seed": 0,
                "cfg_scale_text": 2.5,
                "cfg_scale_caption": 3.5,
                "cfg_scale_speaker": 4.5,
                "cfg_guidance_mode": "joint",
                "cfg_min_t": 0.25,
                "cfg_max_t": 0.75,
                "max_reference_seconds": 10.0,
                "no_context_kv_cache": True,
            },
        )
    )

    assert result.audio == b"RIFFfakeWAVE"
    assert result.media_type == "audio/wav"
    assert len(FakeMLXRuntime.instances) == 1
    assert manager.status_metadata()["loaded"] is True
    runtime = FakeMLXRuntime.instances[0]
    assert runtime.config.weights_path == "/weights/model.npz"
    assert runtime.config.text_tokenizer_repo == "text-tokenizer"
    assert runtime.config.caption_tokenizer_repo == "caption-tokenizer"
    assert runtime.config.text_max_length == 128
    assert runtime.config.caption_max_length == 64
    assert runtime.requests[0] == FakeGenerationRequest(
        text="hello",
        output_wav=runtime.requests[0].output_wav,
        reference_wav=None,
        no_reference=True,
        caption="warm voice",
        seconds=1.5,
        duration_scale=0.5,
        num_steps=12,
        cfg_scale_text=2.5,
        cfg_scale_caption=3.5,
        cfg_scale_speaker=4.5,
        cfg_guidance_mode="joint",
        cfg_min_t=0.25,
        cfg_max_t=0.75,
        seed=0,
        max_reference_seconds=10.0,
        use_context_kv_cache=False,
    )


def test_mlx_runtime_manager_maps_voicedesign_preset_to_runtime_steps() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={
                "caption": "calm narration, clear diction",
                "no_reference": True,
                "preset": "balanced",
                "cfg_scale_caption": 3.5,
            },
        )
    )

    request = FakeMLXRuntime.instances[0].requests[0]
    assert request.caption == "calm narration, clear diction"
    assert request.no_reference is True
    assert request.reference_wav is None
    assert request.num_steps == 24
    assert request.cfg_scale_caption == 3.5


def test_mlx_runtime_manager_lets_explicit_num_steps_override_preset() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_reference": True, "preset": "fast", "num_steps": 32},
        )
    )

    assert FakeMLXRuntime.instances[0].requests[0].num_steps == 32


def test_mlx_runtime_manager_ignores_empty_lora_adapter_option() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_reference": True, "lora_adapter": ""},
        )
    )

    assert FakeMLXRuntime.instances[0].requests[0].text == "hello"


def test_mlx_runtime_manager_rejects_lora_adapter_alias_until_runtime_supports_it() -> None:
    module_loader_called = False

    def module_loader():
        nonlocal module_loader_called
        module_loader_called = True
        return fake_module_loader()

    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="lora_adapter is not supported"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="voicedesign",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": True, "lora_adapter": "warm-narration"},
            )
        )
    assert module_loader_called is False


def test_mlx_runtime_manager_rejects_path_like_lora_adapter_values() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="arbitrary local paths are not accepted"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="voicedesign",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": True, "lora_adapter": "../adapters/warm.safetensors"},
            )
        )


def test_mlx_runtime_manager_rejects_non_string_lora_adapter_values() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="lora_adapter must be a string"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="voicedesign",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": True, "lora_adapter": 123},
            )
        )


def test_split_text_for_generation_prefers_punctuation_boundaries() -> None:
    assert split_text_for_generation("短い文です。次の文です。最後です。", max_chars=8) == [
        "短い文です。",
        "次の文です。",
        "最後です。",
    ]


def test_split_text_for_generation_preserves_spaces_between_punctuation_segments() -> None:
    assert split_text_for_generation("one. two. three.", max_chars=10) == [
        "one. two.",
        "three.",
    ]


def test_split_text_for_generation_preserves_missing_spaces_between_punctuation_segments() -> None:
    assert split_text_for_generation("文。次。終。", max_chars=4) == [
        "文。次。",
        "終。",
    ]


def test_split_text_for_generation_preserves_unsplit_input_spacing() -> None:
    assert split_text_for_generation("  hello  ", max_chars=20) == ["  hello  "]
    assert split_text_for_generation("   ", max_chars=5) == ["   "]


def test_split_text_for_generation_keeps_whitespace_only_chunks_non_empty() -> None:
    assert split_text_for_generation("      ", max_chars=2) == ["  ", "  ", "  "]


def test_split_text_for_generation_falls_back_to_hard_slices() -> None:
    assert split_text_for_generation("abcdefghijklmnopqrstuvwxyz", max_chars=10) == [
        "abcdefghij",
        "klmnopqrst",
        "uvwxyz",
    ]


def test_mlx_runtime_manager_chunks_long_text_and_concatenates_wav_output() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            text_max_length=8,
        ),
        runtime_factory=WaveMLXRuntime,
        module_loader=fake_module_loader,
    )

    result = manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello. goodbye.",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_reference": True},
        )
    )

    requests = WaveMLXRuntime.instances[0].requests
    assert [request.text for request in requests] == ["hello.", "goodbye."]
    assert [request.seconds for request in requests] == [None, None]
    assert read_wav_samples(result.audio) == [6] * 20 + [0] * 10 + [8] * 20 + [0] * 10


def test_mlx_runtime_manager_distributes_explicit_seconds_across_chunks() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            text_max_length=5,
        ),
        runtime_factory=WaveMLXRuntime,
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="abcde fghij",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_reference": True, "seconds": 4.0},
        )
    )

    assert [request.text for request in WaveMLXRuntime.instances[0].requests] == ["abcde", "fghij"]
    assert [request.seconds for request in WaveMLXRuntime.instances[0].requests] == [2.0, 2.0]


def test_mlx_runtime_manager_can_disable_chunking() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            text_max_length=5,
        ),
        runtime_factory=WaveMLXRuntime,
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="abcde fghij",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_reference": True, "chunking": False},
        )
    )

    assert [request.text for request in WaveMLXRuntime.instances[0].requests] == ["abcde fghij"]


def test_mlx_runtime_manager_applies_tail_artifact_controls_per_chunk() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            text_max_length=5,
        ),
        runtime_factory=WaveMLXRuntime,
        module_loader=fake_module_loader,
    )

    result = manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="abcde fghij",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={
                "no_reference": True,
                "tail_trim_ms": 5,
                "tail_silence_trim_ms": 5,
                "tail_silence_keep_ms": 2,
                "tail_silence_threshold": 0,
            },
        )
    )

    assert read_wav_samples(result.audio) == [5] * 20 + [0] * 2 + [5] * 20 + [0] * 2


def test_concatenate_wav_audio_maps_empty_wav_bytes_to_runtime_unavailable() -> None:
    with pytest.raises(RuntimeUnavailableError, match="invalid WAV audio"):
        concatenate_wav_audio([b"", make_wav([1, 2, 3])])


def test_mlx_runtime_manager_synchronizes_lazy_runtime_initialization() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    def generate() -> bytes:
        result = manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
            )
        )
        return result.audio

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: generate(), range(4)))

    assert results == [b"RIFFfakeWAVE"] * 4
    assert len(FakeMLXRuntime.instances) == 1
    assert len(FakeMLXRuntime.instances[0].requests) == 4


def test_mlx_runtime_manager_treats_backend_value_error_as_unavailable() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        runtime_factory=ValueErrorMLXRuntime,
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeUnavailableError, match="backend failed"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
            )
        )


def test_mlx_runtime_manager_uses_hosted_weights_layout() -> None:
    calls = []
    layout = SimpleNamespace(
        weights_path="/layout/model.npz", model_config={"family": "voicedesign"}
    )

    def module_loader():
        runtime_module, _resolver = fake_module_loader()

        def resolve_weights_layout_source(**kwargs):
            calls.append(kwargs)
            return layout

        return runtime_module, resolve_weights_layout_source

    manager = IrodoriMLXRuntimeManager(
        IrodoriRuntimeConfig(weights_repo="owner/repo", weights_revision="main"),
        module_loader=module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.0,
        )
    )

    assert calls == [{"weights_dir": None, "weights_repo": "owner/repo", "revision": "main"}]
    assert FakeMLXRuntime.instances[-1].config.weights_path == "/layout/model.npz"
    assert FakeMLXRuntime.instances[-1].config.model_config == {"family": "voicedesign"}


def test_mlx_runtime_manager_rejects_conflicting_reference_options() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="cannot both be set"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
                irodori={"reference_wav": "/tmp/ref.wav", "no_reference": True},
            )
        )


def test_mlx_runtime_manager_rejects_no_reference_false_without_reference_wav() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="requires irodori.reference_wav"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": False},
            )
        )


def test_mlx_runtime_manager_rejects_invalid_voicedesign_preset() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="irodori.preset must be one of"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="voicedesign",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": True, "preset": "draft"},
            )
        )


def test_mlx_runtime_manager_rejects_invalid_caption_option_type() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="irodori.caption must be a string"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="voicedesign",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": True, "caption": 123},
            )
        )


def test_mlx_runtime_manager_rejects_invalid_cfg_timestep_range() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="irodori.cfg_min_t must be <= irodori.cfg_max_t"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="voicedesign",
                response_format="wav",
                speed=1.0,
                irodori={"no_reference": True, "cfg_min_t": 0.9, "cfg_max_t": 0.1},
            )
        )


def test_mlx_runtime_manager_reports_missing_dependencies_clearly() -> None:
    def missing_module_loader():
        raise RuntimeUnavailableError("Irodori-TTS-MLX runtime dependencies are not installed.")

    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=missing_module_loader,
    )

    with pytest.raises(RuntimeUnavailableError, match="runtime dependencies are not installed"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
            )
        )
    assert "runtime dependencies are not installed" in manager.status_metadata()["last_load_error"]


def test_create_default_runtime_reports_preload_failure_as_configuration_error(monkeypatch) -> None:
    import irodori_tts_mlx_server.runtime as runtime_module

    class FailingRuntimeManager:
        def __init__(self, config: IrodoriRuntimeConfig) -> None:
            assert config.preload is True
            raise RuntimeUnavailableError("preload failed")

    monkeypatch.setattr(runtime_module, "IrodoriMLXRuntimeManager", FailingRuntimeManager)

    runtime = create_default_runtime(
        IrodoriRuntimeConfig(model_id="custom-model", weights_repo="owner/repo", preload=True)
    )

    assert runtime.list_models() == ["custom-model"]
    assert runtime.status_metadata() == {
        "runtime": "configuration_error",
        "configured": False,
        "loaded": False,
        "load_state": "failed",
        "model_id": "custom-model",
        "last_load_error": "preload failed",
    }
    with pytest.raises(RuntimeUnavailableError, match="preload failed"):
        runtime.generate_speech(
            SpeechGenerationRequest(
                model="custom-model",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
            )
        )


def test_mlx_runtime_manager_reports_loading_and_failed_states() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_module_loader():
        started.set()
        if not release.wait(timeout=2):
            raise AssertionError("test did not release runtime loader")
        raise RuntimeUnavailableError("load failed")

    manager = IrodoriMLXRuntimeManager(
        IrodoriRuntimeConfig(weights_repo="owner/repo"),
        module_loader=blocking_module_loader,
    )
    failure: list[str] = []

    def generate() -> None:
        try:
            manager.generate_speech(
                SpeechGenerationRequest(
                    model="irodori-tts-mlx",
                    input="hello",
                    voice="alloy",
                    response_format="wav",
                    speed=1.0,
                )
            )
        except RuntimeUnavailableError as exc:
            failure.append(str(exc))

    thread = threading.Thread(target=generate)
    thread.start()
    assert started.wait(timeout=2)
    try:
        loading_status = manager.status_metadata()
        assert loading_status["loaded"] is False
        assert loading_status["load_state"] == "loading"
    finally:
        release.set()
        thread.join(timeout=2)

    assert failure == ["load failed"]
    failed_status = manager.status_metadata()
    assert failed_status["loaded"] is False
    assert failed_status["load_state"] == "failed"
    assert failed_status["last_load_error"] == "load failed"

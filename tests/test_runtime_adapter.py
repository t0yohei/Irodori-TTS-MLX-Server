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
    ref_wav: str | None = None
    no_ref: bool = False
    caption: str | None = None
    seconds: float | None = None
    duration_scale: float = 1.0
    max_auto_seconds: float | None = None
    max_auto_estimate_seconds: float | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    seed: int = 0
    max_ref_seconds: float | None = 30.0
    context_kv_cache: bool = True


@dataclass(frozen=True)
class FakeRuntimeConfig:
    model_config: object
    weights_path: str
    text_tokenizer_repo: str | None
    caption_tokenizer_repo: str | None
    max_text_len: int
    max_caption_len: int | None
    codec: object


@dataclass(frozen=True)
class FakeCodecConfig:
    codec_repo: str
    codec_path: str | None
    codec_device: str
    runtime_mode: str
    enable_watermark: bool
    normalize_db: float | None


@dataclass(frozen=True)
class FakeUnsupportedCodecConfig:
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


class CountingReferenceBridge:
    def __init__(self) -> None:
        self.encode_calls: list[dict[str, object]] = []

    def encode_reference(
        self,
        path: str,
        *,
        max_seconds: float | None,
        normalize_db: float | None,
        ensure_max: bool,
    ) -> object:
        self.encode_calls.append(
            {
                "path": path,
                "max_seconds": max_seconds,
                "normalize_db": normalize_db,
                "ensure_max": ensure_max,
            }
        )
        return object()


class ReferenceEncodingMLXRuntime:
    instances: list["ReferenceEncodingMLXRuntime"] = []

    def __init__(self, *, config: FakeRuntimeConfig) -> None:
        self.config = config
        self.bridge = CountingReferenceBridge()
        self.raw_bridge = self.bridge
        self.requests: list[FakeGenerationRequest] = []
        ReferenceEncodingMLXRuntime.instances.append(self)

    def generate(self, request: FakeGenerationRequest) -> object:
        self.requests.append(request)
        assert request.ref_wav is not None
        self.bridge.encode_reference(
            request.ref_wav,
            max_seconds=request.max_ref_seconds,
            normalize_db=-16.0,
            ensure_max=True,
        )
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


def fake_codec_resolver(**_kwargs):
    return SimpleNamespace(
        codec_path="/resolved-codec/dacvae-codec.npz",
        source="t0yohei/Irodori-TTS-MLX-DACVAE-Codec",
        source_kind="repo",
    )


def fake_module_loader():
    runtime_module = SimpleNamespace(
        DACVAEBridgeConfig=FakeCodecConfig,
        SamplingRequest=FakeGenerationRequest,
        InferenceRuntime=FakeMLXRuntime,
        MLXRuntimeConfig=FakeRuntimeConfig,
    )
    layout = SimpleNamespace(
        weights_path="/weights/model.npz",
        model_config={"family": "voicedesign"},
    )
    return runtime_module, lambda **_kwargs: layout, fake_codec_resolver


def fake_unsupported_codec_module_loader():
    runtime_module, resolver, codec_resolver = fake_module_loader()
    runtime_module.DACVAEBridgeConfig = FakeUnsupportedCodecConfig
    return runtime_module, resolver, codec_resolver


def fake_missing_inference_runtime_module_loader():
    runtime_module, resolver, codec_resolver = fake_module_loader()
    del runtime_module.InferenceRuntime
    return runtime_module, resolver, codec_resolver


def fake_missing_sampling_request_module_loader():
    runtime_module, resolver, codec_resolver = fake_module_loader()
    del runtime_module.SamplingRequest
    return runtime_module, resolver, codec_resolver


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
            max_text_len=128,
            max_caption_len=64,
            codec_path="/codec/semantic-dacvae-mlx.npz",
            codec_device="cpu",
            codec_runtime_mode="mlx-decode",
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
                "no_ref": True,
                "seconds": 1.5,
                "num_steps": 12,
                "seed": 0,
                "cfg_scale_text": 2.5,
                "cfg_scale_caption": 3.5,
                "cfg_scale_speaker": 4.5,
                "cfg_guidance_mode": "joint",
                "cfg_min_t": 0.25,
                "cfg_max_t": 0.75,
                "max_ref_seconds": 10.0,
                "context_kv_cache": False,
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
    assert runtime.config.max_text_len == 128
    assert runtime.config.max_caption_len == 64
    assert runtime.config.codec == FakeCodecConfig(
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        codec_path="/codec/semantic-dacvae-mlx.npz",
        codec_device="cpu",
        runtime_mode="mlx-decode",
        enable_watermark=False,
        normalize_db=-16.0,
    )
    assert runtime.requests[0] == FakeGenerationRequest(
        text="hello",
        output_wav=runtime.requests[0].output_wav,
        ref_wav=None,
        no_ref=True,
        caption="warm voice",
        seconds=1.5,
        duration_scale=0.5,
        max_auto_seconds=None,
        max_auto_estimate_seconds=None,
        num_steps=12,
        cfg_scale_text=2.5,
        cfg_scale_caption=3.5,
        cfg_scale_speaker=4.5,
        cfg_guidance_mode="joint",
        cfg_min_t=0.25,
        cfg_max_t=0.75,
        seed=0,
        max_ref_seconds=10.0,
        context_kv_cache=False,
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
                "no_ref": True,
                "preset": "balanced",
                "cfg_scale_caption": 3.5,
            },
        )
    )

    request = FakeMLXRuntime.instances[0].requests[0]
    assert request.caption == "calm narration, clear diction"
    assert request.no_ref is True
    assert request.ref_wav is None
    assert request.num_steps == 24
    assert request.cfg_scale_caption == 3.5


def test_mlx_runtime_manager_maps_ultra_fast_preset_to_runtime_policy() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.0,
            irodori={"no_ref": True, "preset": "ultra-fast"},
        )
    )

    request = FakeMLXRuntime.instances[0].requests[0]
    assert request.num_steps == 8
    assert request.max_auto_seconds == 2.5
    assert request.max_auto_estimate_seconds == 3.0


def test_mlx_runtime_manager_skips_ultra_fast_cap_when_duration_scale_is_explicit() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.0,
            irodori={
                "no_ref": True,
                "preset": "ultra-fast",
                "duration_scale": 1.25,
            },
        )
    )

    request = FakeMLXRuntime.instances[0].requests[0]
    assert request.num_steps == 8
    assert request.duration_scale == 1.25
    assert request.max_auto_seconds is None
    assert request.max_auto_estimate_seconds is None


def test_mlx_runtime_manager_skips_ultra_fast_cap_when_seconds_is_explicit_null() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.0,
            irodori={
                "no_ref": True,
                "preset": "ultra-fast",
                "seconds": None,
            },
        )
    )

    request = FakeMLXRuntime.instances[0].requests[0]
    assert request.num_steps == 8
    assert request.seconds is None
    assert request.max_auto_seconds is None
    assert request.max_auto_estimate_seconds is None


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
            irodori={"no_ref": True, "preset": "fast", "num_steps": 32},
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
            irodori={"no_ref": True, "lora_adapter": ""},
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
                irodori={"no_ref": True, "lora_adapter": "warm-narration"},
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
                irodori={"no_ref": True, "lora_adapter": "../adapters/warm.safetensors"},
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
                irodori={"no_ref": True, "lora_adapter": 123},
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


def test_split_text_for_generation_punctuation_mode_merges_short_segments() -> None:
    assert split_text_for_generation(
        "こんにちは。これは stream chunks の動作確認です。最初の音声が返ったら、"
        "続きの音声を生成しながら再生できます。",
        max_chars=256,
        chunk_mode="punctuation",
    ) == [
        "こんにちは。",
        "これは stream chunks の動作確認です。",
        "最初の音声が返ったら、続きの音声を生成しながら再生できます。",
    ]


def test_split_text_for_generation_first_sentence_comma_chunking() -> None:
    assert split_text_for_generation(
        "最初は速く、すぐ返します。次は長くて、通常のままです。",
        max_chars=256,
        chunk_mode="punctuation",
        first_sentence_comma_chunking_enabled=True,
    ) == [
        "最初は速く、",
        "すぐ返します。",
        "次は長くて、通常のままです。",
    ]


def test_first_sentence_comma_chunking_leaves_text_without_comma_unchanged() -> None:
    assert split_text_for_generation(
        "最初の文です。次は長くて、通常のままです。",
        max_chars=256,
        chunk_mode="punctuation",
        first_sentence_comma_chunking_enabled=True,
    ) == [
        "最初の文です。",
        "次は長くて、通常のままです。",
    ]


def test_split_text_for_generation_punctuation_mode_hard_splits_unbroken_text() -> None:
    assert split_text_for_generation(
        "abcdefghijklmnopqrstuvwxyz",
        max_chars=10,
        chunk_mode="punctuation",
    ) == [
        "abcdefghij",
        "klmnopqrst",
        "uvwxyz",
    ]


def test_split_text_for_generation_punctuation_mode_preserves_whitespace_only_chunks() -> None:
    assert split_text_for_generation(
        "      ",
        max_chars=2,
        chunk_mode="punctuation",
    ) == ["  ", "  ", "  "]


def test_punctuation_mode_drops_boundary_whitespace_after_period() -> None:
    assert split_text_for_generation(
        "こんにちは。  次です。\n最後です。",
        max_chars=256,
        chunk_mode="punctuation",
    ) == ["こんにちは。", "次です。", "最後です。"]


def test_split_text_for_generation_punctuation_mode_keeps_closers_with_period() -> None:
    assert split_text_for_generation(
        '"こんにちは。"次です。終わり。\'',
        max_chars=256,
        chunk_mode="punctuation",
    ) == ['"こんにちは。"', "次です。", "終わり。'"]


def test_split_text_for_generation_punctuation_mode_keeps_overflow_closers_attached() -> None:
    assert split_text_for_generation(
        'ab。"c',
        max_chars=3,
        chunk_mode="punctuation",
        chunk_hard_max_chars=3,
    ) == ['ab。"', "c"]


def test_split_text_for_generation_punctuation_mode_keeps_terminal_marks_with_period() -> None:
    assert split_text_for_generation(
        "こんにちは。！？次です。",
        max_chars=256,
        chunk_mode="punctuation",
    ) == ["こんにちは。！？", "次です。"]


def test_mlx_runtime_manager_chunks_long_text_and_concatenates_wav_output() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            max_text_len=8,
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
            irodori={"no_ref": True},
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
            max_text_len=5,
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
            irodori={"no_ref": True, "seconds": 4.0},
        )
    )

    assert [request.text for request in WaveMLXRuntime.instances[0].requests] == ["abcde", "fghij"]
    assert [request.seconds for request in WaveMLXRuntime.instances[0].requests] == [2.0, 2.0]


def test_mlx_runtime_manager_supports_punctuation_chunking_enabled() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            max_text_len=256,
        ),
        runtime_factory=WaveMLXRuntime,
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input=(
                "こんにちは。これは stream chunks の動作確認です。"
                "最初の音声が返ったら、続きの音声を生成しながら再生できます。"
            ),
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_ref": True, "punctuation_chunking_enabled": True},
        )
    )

    assert [request.text for request in WaveMLXRuntime.instances[0].requests] == [
        "こんにちは。",
        "これは stream chunks の動作確認です。",
        "最初の音声が返ったら、続きの音声を生成しながら再生できます。",
    ]


def test_mlx_runtime_manager_clamps_punctuation_chunks_to_text_max_length() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            max_text_len=8,
        ),
        runtime_factory=WaveMLXRuntime,
        module_loader=fake_module_loader,
    )

    manager.generate_speech(
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="abcdefghijklmnopqrstuvwxyz",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={"no_ref": True, "punctuation_chunking_enabled": True},
        )
    )

    assert [request.text for request in WaveMLXRuntime.instances[0].requests] == [
        "abcdefgh",
        "ijklmnop",
        "qrstuvwx",
        "yz",
    ]


def test_mlx_runtime_manager_can_disable_chunking() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            max_text_len=5,
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
            irodori={"no_ref": True, "chunking_enabled": False},
        )
    )

    assert [request.text for request in WaveMLXRuntime.instances[0].requests] == ["abcde fghij"]


def test_mlx_runtime_manager_applies_tail_artifact_controls_per_chunk() -> None:
    WaveMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(
            max_text_len=5,
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
                "no_ref": True,
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
        runtime_module, _resolver, codec_resolver = fake_module_loader()

        def resolve_weights_layout_source(**kwargs):
            calls.append(kwargs)
            return layout

        return runtime_module, resolve_weights_layout_source, codec_resolver

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


def test_mlx_runtime_manager_resolves_default_hosted_codec_repo_for_default_mlx_mode() -> None:
    FakeMLXRuntime.instances.clear()
    calls = []

    def module_loader():
        runtime_module, resolver, _codec_resolver = fake_module_loader()

        def resolve_codec_artifact_source(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                codec_path="/hf-cache/dacvae-codec.npz",
                source="t0yohei/Irodori-TTS-MLX-DACVAE-Codec",
                source_kind="repo",
            )

        return runtime_module, resolver, resolve_codec_artifact_source

    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
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

    assert calls == [
        {
            "codec_artifact_repo": "t0yohei/Irodori-TTS-MLX-DACVAE-Codec",
            "revision": None,
        }
    ]
    assert FakeMLXRuntime.instances[-1].config.codec.codec_path == "/hf-cache/dacvae-codec.npz"
    assert FakeMLXRuntime.instances[-1].config.codec.runtime_mode == "mlx"


def managed_reference_cache_options(
    *,
    voice_id: str = "sample",
    path: str = "/voices/sample.wav",
    size: int = 4,
    mtime_ns: int = 100,
) -> dict[str, object]:
    return {
        "_managed_reference_cache": {
            "voice_id": voice_id,
            "path": path,
            "size": size,
            "mtime_ns": mtime_ns,
        }
    }


def reference_request(**irodori_overrides: object) -> SpeechGenerationRequest:
    cache_info = irodori_overrides.get("_managed_reference_cache")
    ref_wav = (
        cache_info["path"]
        if isinstance(cache_info, dict) and isinstance(cache_info.get("path"), str)
        else "/voices/sample.wav"
    )
    irodori = {
        "ref_wav": ref_wav,
        "no_ref": False,
    }
    irodori.update(irodori_overrides)
    return SpeechGenerationRequest(
        model="irodori-tts-mlx",
        input="hello",
        voice="sample",
        response_format="wav",
        speed=1.0,
        irodori=irodori,
    )


def test_mlx_runtime_manager_caches_managed_reference_encoding() -> None:
    ReferenceEncodingMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        runtime_factory=ReferenceEncodingMLXRuntime,
        module_loader=fake_module_loader,
    )
    manager.configure_managed_reference_cache(max_entries=2)
    request = reference_request(**managed_reference_cache_options())

    manager.generate_speech(request)
    manager.generate_speech(request)

    runtime = ReferenceEncodingMLXRuntime.instances[-1]
    assert len(runtime.raw_bridge.encode_calls) == 1
    cache = manager.status_metadata()["managed_reference_cache"]
    assert cache["hits"] == 1
    assert cache["misses"] == 1


def test_mlx_runtime_manager_cache_misses_when_managed_voice_file_changes() -> None:
    ReferenceEncodingMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        runtime_factory=ReferenceEncodingMLXRuntime,
        module_loader=fake_module_loader,
    )
    manager.configure_managed_reference_cache(max_entries=2)

    manager.generate_speech(reference_request(**managed_reference_cache_options(mtime_ns=100)))
    manager.generate_speech(reference_request(**managed_reference_cache_options(mtime_ns=200)))

    runtime = ReferenceEncodingMLXRuntime.instances[-1]
    assert len(runtime.raw_bridge.encode_calls) == 2
    assert manager.status_metadata()["managed_reference_cache"]["misses"] == 2


def test_mlx_runtime_manager_invalidates_managed_reference_cache_by_voice_id() -> None:
    ReferenceEncodingMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        runtime_factory=ReferenceEncodingMLXRuntime,
        module_loader=fake_module_loader,
    )
    manager.configure_managed_reference_cache(max_entries=2)
    request = reference_request(**managed_reference_cache_options())

    manager.generate_speech(request)
    manager.invalidate_managed_reference_cache("sample")
    manager.generate_speech(request)

    runtime = ReferenceEncodingMLXRuntime.instances[-1]
    assert len(runtime.raw_bridge.encode_calls) == 2


def test_mlx_runtime_manager_managed_reference_cache_is_bounded_and_disableable() -> None:
    ReferenceEncodingMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        runtime_factory=ReferenceEncodingMLXRuntime,
        module_loader=fake_module_loader,
    )
    manager.configure_managed_reference_cache(max_entries=1)

    manager.generate_speech(
        reference_request(**managed_reference_cache_options(voice_id="a", path="/voices/a.wav"))
    )
    manager.generate_speech(
        reference_request(**managed_reference_cache_options(voice_id="b", path="/voices/b.wav"))
    )
    manager.generate_speech(
        reference_request(**managed_reference_cache_options(voice_id="a", path="/voices/a.wav"))
    )

    runtime = ReferenceEncodingMLXRuntime.instances[-1]
    assert len(runtime.raw_bridge.encode_calls) == 3
    assert manager.status_metadata()["managed_reference_cache"]["evictions"] == 2

    manager.configure_managed_reference_cache(max_entries=0)
    disabled_request = reference_request(**managed_reference_cache_options())
    manager.generate_speech(disabled_request)
    manager.generate_speech(disabled_request)

    assert len(runtime.raw_bridge.encode_calls) == 5
    assert manager.status_metadata()["managed_reference_cache"]["enabled"] is False
    assert manager.status_metadata()["codec_artifact_source"] == (
        "t0yohei/Irodori-TTS-MLX-DACVAE-Codec"
    )
    assert manager.status_metadata()["codec_artifact_source_kind"] == "repo"


def test_mlx_runtime_manager_prefers_explicit_codec_path_over_hosted_codec_repo() -> None:
    FakeMLXRuntime.instances.clear()
    calls = []

    def module_loader():
        runtime_module, resolver, _codec_resolver = fake_module_loader()

        def resolve_codec_artifact_source(**kwargs):
            calls.append(kwargs)
            return fake_codec_resolver(**kwargs)

        return runtime_module, resolver, resolve_codec_artifact_source

    manager = IrodoriMLXRuntimeManager(
        hosted_config(codec_path="/codec/local.npz", codec_runtime_mode="mlx-decode"),
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

    assert calls == []
    assert FakeMLXRuntime.instances[-1].config.codec.codec_path == "/codec/local.npz"
    assert manager.status_metadata()["codec_artifact_source"] == "/codec/local.npz"
    assert manager.status_metadata()["codec_artifact_source_kind"] == "path"


def test_mlx_runtime_manager_uses_codec_path_for_default_mlx_mode() -> None:
    FakeMLXRuntime.instances.clear()
    manager = IrodoriMLXRuntimeManager(
        hosted_config(codec_path="/codec/local.npz"),
        module_loader=fake_module_loader,
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

    assert FakeMLXRuntime.instances[-1].config.codec.codec_path == "/codec/local.npz"
    assert FakeMLXRuntime.instances[-1].config.codec.runtime_mode == "mlx"
    assert manager.status_metadata()["codec_artifact_source"] == "/codec/local.npz"
    assert manager.status_metadata()["codec_artifact_source_kind"] == "path"


def test_mlx_runtime_manager_rejects_non_mlx_codec_runtime_mode() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(codec_runtime_mode="legacy"),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeUnavailableError, match="Unsupported MLX codec runtime mode"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
            )
        )


def test_mlx_runtime_manager_rejects_runtime_without_mlx_codec_artifact_support() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_unsupported_codec_module_loader,
    )

    with pytest.raises(RuntimeUnavailableError, match="does not support MLX DACVAE codec"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
            )
        )


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
                irodori={"ref_wav": "/tmp/ref.wav", "no_ref": True},
            )
        )


def test_mlx_runtime_manager_rejects_no_ref_false_without_ref_wav() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=fake_module_loader,
    )

    with pytest.raises(RuntimeRequestError, match="requires irodori.ref_wav"):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
                irodori={"no_ref": False},
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
                irodori={"no_ref": True, "preset": "draft"},
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
                irodori={"no_ref": True, "caption": 123},
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
                irodori={"no_ref": True, "cfg_min_t": 0.9, "cfg_max_t": 0.1},
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


@pytest.mark.parametrize(
    ("module_loader", "missing_symbol"),
    [
        (fake_missing_inference_runtime_module_loader, "InferenceRuntime"),
        (fake_missing_sampling_request_module_loader, "SamplingRequest"),
    ],
)
def test_mlx_runtime_manager_reports_missing_runtime_api_symbols(
    module_loader, missing_symbol
) -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(),
        module_loader=module_loader,
    )

    with pytest.raises(RuntimeUnavailableError, match=missing_symbol):
        manager.generate_speech(
            SpeechGenerationRequest(
                model="irodori-tts-mlx",
                input="hello",
                voice="alloy",
                response_format="wav",
                speed=1.0,
                irodori={"no_ref": True},
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


def test_mlx_runtime_manager_status_reports_codec_configuration() -> None:
    manager = IrodoriMLXRuntimeManager(
        hosted_config(codec_path="/codec/semantic-dacvae-mlx.npz", codec_runtime_mode="mlx"),
        module_loader=fake_module_loader,
    )

    assert manager.status_metadata() == {
        "runtime": "irodori-tts-mlx",
        "configured": True,
        "loaded": False,
        "load_state": "not_loaded",
        "model_id": "irodori-tts-mlx",
        "weights_source": "weights_repo",
        "codec_path_configured": True,
        "codec_artifact_repo": "t0yohei/Irodori-TTS-MLX-DACVAE-Codec",
        "codec_artifact_revision": None,
        "codec_artifact_source": None,
        "codec_artifact_source_kind": None,
        "codec_runtime_mode": "mlx",
        "last_load_error": None,
        "managed_reference_cache": {
            "enabled": True,
            "max_entries": 8,
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        },
    }

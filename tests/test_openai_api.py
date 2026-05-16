import importlib
import io
import wave

from fastapi.testclient import TestClient

from irodori_tts_mlx_server import create_app
from irodori_tts_mlx_server.runtime import (
    RuntimeRequestError,
    SpeechGenerationRequest,
    SpeechGenerationResult,
)


def wav_bytes(pcm: bytes = bytes([1, 2, 3, 4])) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class MockSpeechRuntime:
    def __init__(self) -> None:
        self.requests: list[SpeechGenerationRequest] = []

    def list_models(self) -> list[str]:
        return ["irodori-tts-mlx"]

    def status_metadata(self) -> dict[str, object]:
        return {"runtime": "mock", "configured": True, "loaded": True}

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        self.requests.append(request)
        return SpeechGenerationResult(audio=wav_bytes(), media_type="audio/wav")


class FalseyMockSpeechRuntime(MockSpeechRuntime):
    def __len__(self) -> int:
        return 0


class InvalidOptionsRuntime(MockSpeechRuntime):
    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeRequestError("irodori.num_steps must be > 0.")


def test_v1_models_returns_openai_compatible_model_list() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "irodori-tts-mlx",
                "object": "model",
                "created": 0,
                "owned_by": "irodori-tts-mlx",
            }
        ],
    }


def test_audio_speech_returns_complete_wav_bytes_from_runtime() -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(create_app(runtime=runtime))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.25,
            "irodori": {"speaker_id": 1},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == wav_bytes()
    assert runtime.requests == [
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.25,
            irodori={"speaker_id": 1},
        )
    ]


def test_audio_speech_accepts_voicedesign_caption_no_reference_options() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "おはようございます",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {
                "caption": "落ち着いたナレーション。明瞭で少し低めの声。",
                "no_reference": True,
                "preset": "balanced",
                "cfg_scale_caption": 3.5,
                "seconds": 2.0,
                "seed": 1234,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.requests == [
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="おはようございます",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={
                "caption": "落ち着いたナレーション。明瞭で少し低めの声。",
                "no_reference": True,
                "preset": "balanced",
                "cfg_scale_caption": 3.5,
                "seconds": 2.0,
                "seed": 1234,
            },
        )
    ]


def test_audio_speech_accepts_chunking_and_tail_artifact_controls() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello. goodbye.",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {
                "no_reference": True,
                "chunking": True,
                "chunk_max_chars": 8,
                "tail_trim_ms": 20,
                "tail_silence_trim_ms": 120,
                "tail_silence_keep_ms": 40,
                "tail_silence_threshold": 512,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori == {
        "no_reference": True,
        "chunking": True,
        "chunk_max_chars": 8,
        "tail_trim_ms": 20,
        "tail_silence_trim_ms": 120,
        "tail_silence_keep_ms": 40,
        "tail_silence_threshold": 512,
    }


def test_falsey_runtime_injection_is_preserved() -> None:
    runtime = FalseyMockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "irodori-tts-mlx"


def test_audio_speech_offloads_generation_and_conversion_to_threadpool(monkeypatch) -> None:
    app_module = importlib.import_module("irodori_tts_mlx_server.factory")
    calls = []

    async def fake_run_in_threadpool(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(app_module, "run_in_threadpool", fake_run_in_threadpool)
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0][0] == runtime.generate_speech
    assert calls[1][0] == app_module.convert_audio_response


def test_audio_speech_rejects_streaming_requests() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Streaming audio responses and SSE are not supported; request complete audio bytes.",
        "type": "invalid_request_error",
        "param": "stream",
        "code": "unsupported_streaming",
    }


def test_audio_speech_rejects_sse_accept_header() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        headers={"accept": "text/event-stream"},
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_streaming"


def test_audio_speech_rejects_unknown_model() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "missing-model",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "message": "Model 'missing-model' is not available.",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found",
    }


def test_audio_speech_validation_errors_use_openai_error_shape() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == "input"
    assert response.json()["error"]["code"] == "validation_error"


def test_audio_speech_rejects_unsupported_response_format() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "ogg",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == "response_format"
    assert response.json()["error"]["code"] == "validation_error"


def test_audio_speech_returns_pcm_extracted_from_generated_wav() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "pcm",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.content == bytes([1, 2, 3, 4])
    assert runtime.requests == [
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="hello",
            voice="alloy",
            response_format="wav",
            speed=1.0,
            irodori={},
        )
    ]


def test_audio_speech_reports_missing_ffmpeg_for_compressed_default_format(monkeypatch) -> None:
    app_module = importlib.import_module("irodori_tts_mlx_server.audio")
    monkeypatch.setattr(app_module.shutil, "which", lambda _name: None)

    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
        },
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": (
            "response_format='mp3' requires FFmpeg. Install FFmpeg or request "
            "response_format='wav' or 'pcm'."
        ),
        "type": "server_error",
        "param": "response_format",
        "code": "response_format_unavailable",
    }
    assert runtime.requests == []


def test_audio_speech_rejects_invalid_irodori_runtime_options() -> None:
    response = TestClient(create_app(runtime=InvalidOptionsRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "irodori": {"num_steps": 0},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "irodori.num_steps must be > 0.",
        "type": "invalid_request_error",
        "param": "irodori",
        "code": "invalid_irodori_options",
    }


def test_audio_speech_preserves_openai_default_format_and_encodes_when_ffmpeg_available(
    monkeypatch,
) -> None:
    app_module = importlib.import_module("irodori_tts_mlx_server.audio")

    def fake_run(command, *, capture_output, text, check):
        assert command[0] == "/usr/bin/ffmpeg"
        assert "-f" in command
        assert command[command.index("-f") + 1] == "mp3"
        output_path = command[-1]
        with open(output_path, "wb") as output_file:
            output_file.write(b"encoded-mp3")

        class Completed:
            returncode = 0
            stderr = ""

        return Completed()

    monkeypatch.setattr(app_module.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"encoded-mp3"
    assert runtime.requests[0].response_format == "wav"


def test_default_app_imports_without_model_weights_and_reports_runtime_unavailable() -> None:
    from irodori_tts_mlx_server import app

    client = TestClient(app)
    models_response = client.get("/v1/models")
    speech_response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert models_response.status_code == 200
    assert speech_response.status_code == 503
    assert speech_response.json()["error"]["code"] == "runtime_unavailable"


def test_default_app_reports_invalid_integer_env_as_runtime_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("IRODORI_MLX_WEIGHTS_REPO", "owner/repo")
    monkeypatch.setenv("IRODORI_MLX_TEXT_MAX_LENGTH", "not-an-int")

    client = TestClient(create_app())
    health_response = client.get("/health")
    speech_response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert health_response.status_code == 200
    assert health_response.json()["speech_runtime"] == {
        "runtime": "configuration_error",
        "configured": False,
        "loaded": False,
        "model_id": "irodori-tts-mlx",
        "last_load_error": "IRODORI_MLX_TEXT_MAX_LENGTH must be an integer.",
    }
    assert speech_response.status_code == 503
    assert speech_response.json()["error"]["code"] == "runtime_unavailable"
    assert "IRODORI_MLX_TEXT_MAX_LENGTH" in speech_response.json()["error"]["message"]

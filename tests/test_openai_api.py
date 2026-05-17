import importlib
import io
import threading
import time
import wave

import pytest
from fastapi.testclient import TestClient

from irodori_tts_mlx_server import create_app
from irodori_tts_mlx_server.config import ServerConfig, server_config_from_env
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


class BlockingSpeechRuntime(MockSpeechRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        self.requests.append(request)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise AssertionError("test did not release blocked synthesis")
        return SpeechGenerationResult(audio=wav_bytes(), media_type="audio/wav")


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


def test_openai_routes_require_bearer_token_when_configured() -> None:
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(bearer_token="secret"))
    )

    missing_response = client.get("/v1/models")
    invalid_response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer wrong"},
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )
    voices_response = client.get("/v1/audio/voices")
    valid_response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    health_response = client.get("/health")

    assert missing_response.status_code == 401
    assert missing_response.json()["error"] == {
        "message": "Missing or invalid bearer token.",
        "type": "authentication_error",
        "param": None,
        "code": "invalid_api_key",
    }
    assert invalid_response.status_code == 401
    assert voices_response.status_code == 401
    assert valid_response.status_code == 200
    assert health_response.status_code == 200
    assert health_response.json()["server"]["auth_enabled"] is True


def test_openai_routes_allow_local_development_without_auth() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).get("/v1/models")

    assert response.status_code == 200


def test_voice_upload_list_get_replace_delete_and_speech_resolution(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(
        create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path / "voices"))
    )

    created = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"old wav", "audio/wav")},
        data={"voice_id": "sample"},
    )
    listed = client.get("/v1/audio/voices")
    fetched = client.get("/v1/audio/voices/sample")
    speech = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "sample",
            "response_format": "wav",
        },
    )
    replaced = client.put(
        "/v1/audio/voices/sample",
        files={"file": ("renamed.wav", b"new wav", "audio/wav")},
    )
    deleted = client.delete("/v1/audio/voices/sample")
    missing = client.get("/v1/audio/voices/sample")

    assert created.status_code == 201
    assert created.json()["filename"] == "sample.wav"
    assert listed.status_code == 200
    assert listed.json()["data"] == [
        {
            "id": "sample",
            "object": "voice",
            "ref_wav": str(tmp_path / "voices" / "sample.wav"),
            "ref_latent": None,
            "no_ref": False,
        }
    ]
    assert fetched.status_code == 200
    assert speech.status_code == 200
    assert runtime.requests[-1].irodori == {
        "reference_wav": str(tmp_path / "voices" / "sample.wav"),
        "no_reference": False,
    }
    assert replaced.status_code == 200
    assert replaced.json()["bytes"] == len(b"new wav")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": "sample", "object": "voice_file", "deleted": True}
    assert missing.status_code == 404


def test_voice_management_rejects_bad_id_bad_extension_duplicate_and_empty_file(tmp_path) -> None:
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    assert (
        client.post(
            "/v1/audio/voices",
            files={"file": ("sample.wav", b"wav", "audio/wav")},
            data={"voice_id": "sample"},
        ).status_code
        == 201
    )

    duplicate = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )
    bad_id = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "../sample"},
    )
    bad_extension = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.mp3", b"mp3", "audio/mpeg")},
        data={"voice_id": "mp3"},
    )
    empty = client.post(
        "/v1/audio/voices",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"voice_id": "empty"},
    )

    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "voice_exists"
    assert bad_id.status_code == 400
    assert bad_id.json()["error"]["code"] == "invalid_voice"
    assert bad_extension.status_code == 400
    assert bad_extension.json()["error"]["message"] == (
        "Managed reference voices must be uploaded as .wav files."
    )
    assert empty.status_code == 400
    assert empty.json()["error"]["message"] == "Voice file must not be empty."
    assert not (tmp_path.parent / "sample.wav").exists()


def test_managed_voice_does_not_override_explicit_irodori_reference_options(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path)))
    assert (
        client.post(
            "/v1/audio/voices",
            files={"file": ("sample.wav", b"wav", "audio/wav")},
            data={"voice_id": "sample"},
        ).status_code
        == 201
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "sample",
            "response_format": "wav",
            "irodori": {"no_reference": True, "caption": "calm"},
        },
    )

    assert response.status_code == 200
    assert runtime.requests[-1].irodori == {"no_reference": True, "caption": "calm"}


def test_audio_speech_times_out_when_synthesis_queue_is_full() -> None:
    runtime = BlockingSpeechRuntime()
    client = TestClient(
        create_app(
            runtime=runtime,
            config=ServerConfig(max_concurrent_synthesis=1, queue_timeout_seconds=0.01),
        )
    )
    payload = {
        "model": "irodori-tts-mlx",
        "input": "hello",
        "voice": "alloy",
        "response_format": "wav",
    }
    first_status: dict[str, int] = {}

    def first_request() -> None:
        first_status["status_code"] = client.post("/v1/audio/speech", json=payload).status_code

    thread = threading.Thread(target=first_request)
    thread.start()
    assert runtime.started.wait(timeout=2)
    try:
        response = client.post("/v1/audio/speech", json=payload)
    finally:
        runtime.release.set()
        thread.join(timeout=2)

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "Synthesis queue is full or the model is still loading; retry later.",
        "type": "server_error",
        "param": None,
        "code": "synthesis_queue_timeout",
    }
    assert first_status == {"status_code": 200}


def test_audio_speech_allows_zero_queue_timeout_when_slot_is_available() -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(
        create_app(
            runtime=runtime,
            config=ServerConfig(max_concurrent_synthesis=1, queue_timeout_seconds=0),
        )
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == wav_bytes()
    assert len(runtime.requests) == 1


def test_audio_speech_zero_queue_timeout_rejects_when_queue_is_full() -> None:
    runtime = BlockingSpeechRuntime()
    client = TestClient(
        create_app(
            runtime=runtime,
            config=ServerConfig(max_concurrent_synthesis=1, queue_timeout_seconds=0),
        )
    )
    payload = {
        "model": "irodori-tts-mlx",
        "input": "hello",
        "voice": "alloy",
        "response_format": "wav",
    }
    first_status: dict[str, int] = {}

    def first_request() -> None:
        first_status["status_code"] = client.post("/v1/audio/speech", json=payload).status_code

    thread = threading.Thread(target=first_request)
    thread.start()
    assert runtime.started.wait(timeout=2)
    try:
        response = client.post("/v1/audio/speech", json=payload)
    finally:
        runtime.release.set()
        thread.join(timeout=2)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "synthesis_queue_timeout"
    assert first_status == {"status_code": 200}


def test_audio_speech_allows_configured_concurrent_synthesis() -> None:
    class SleepingRuntime(MockSpeechRuntime):
        def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
            self.requests.append(request)
            time.sleep(0.05)
            return SpeechGenerationResult(audio=wav_bytes(), media_type="audio/wav")

    runtime = SleepingRuntime()
    client = TestClient(
        create_app(
            runtime=runtime,
            config=ServerConfig(max_concurrent_synthesis=2, queue_timeout_seconds=1.0),
        )
    )
    payload = {
        "model": "irodori-tts-mlx",
        "input": "hello",
        "voice": "alloy",
        "response_format": "wav",
    }
    statuses: list[int] = []
    threads = [
        threading.Thread(
            target=lambda: statuses.append(
                client.post("/v1/audio/speech", json=payload).status_code
            )
        )
        for _ in range(2)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(statuses) == [200, 200]
    assert len(runtime.requests) == 2


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
    assert health_response.json()["server"]["auth_enabled"] is False
    assert speech_response.status_code == 503
    assert speech_response.json()["error"]["code"] == "runtime_unavailable"
    assert "IRODORI_MLX_TEXT_MAX_LENGTH" in speech_response.json()["error"]["message"]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_server_config_rejects_non_finite_queue_timeout_env(monkeypatch, value: str) -> None:
    monkeypatch.setenv("IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS"):
        server_config_from_env()


def test_invalid_server_env_keeps_health_available_and_blocks_openai_routes(monkeypatch) -> None:
    monkeypatch.setenv("IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS", "nan")

    client = TestClient(create_app(runtime=MockSpeechRuntime()))
    health_response = client.get("/health")
    models_response = client.get("/v1/models")

    assert health_response.status_code == 200
    assert health_response.json()["server"] == {
        "auth_enabled": False,
        "max_concurrent_synthesis": 1,
        "queue_timeout_seconds": 30.0,
        "voices": {
            "dir": "voices",
            "dir_exists": health_response.json()["server"]["voices"]["dir_exists"],
            "files": 0,
            "formats": [".wav"],
        },
        "status": "configuration_error",
        "error": {
            "code": "server_configuration_error",
            "message": "IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS must be a finite number.",
        },
    }
    assert models_response.status_code == 503
    assert models_response.json()["error"] == {
        "message": "IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS must be a finite number.",
        "type": "server_error",
        "param": None,
        "code": "server_configuration_error",
    }

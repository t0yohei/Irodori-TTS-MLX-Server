from fastapi.testclient import TestClient

from irodori_tts_mlx_server import create_app
from irodori_tts_mlx_server.runtime import (
    SpeechGenerationRequest,
    SpeechGenerationResult,
)


class MockSpeechRuntime:
    def __init__(self) -> None:
        self.requests: list[SpeechGenerationRequest] = []

    def list_models(self) -> list[str]:
        return ["irodori-tts-mlx"]

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        self.requests.append(request)
        return SpeechGenerationResult(audio=b"RIFF....WAVEfmt ", media_type="audio/wav")


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
    assert response.content == b"RIFF....WAVEfmt "
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
            "response_format": "mp3",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == "response_format"


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

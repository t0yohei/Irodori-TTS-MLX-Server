import io
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openai import AuthenticationError, BadRequestError, OpenAI

from irodori_tts_mlx_server import create_app
from irodori_tts_mlx_server.config import ServerConfig
from irodori_tts_mlx_server.runtime import (
    RuntimeRequestError,
    SpeechGenerationRequest,
    SpeechGenerationResult,
)


pytestmark = pytest.mark.compatibility


def wav_bytes(pcm: bytes = b"\x01\x02\x03\x04") -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class CompatibilityRuntime:
    def __init__(self) -> None:
        self.requests: list[SpeechGenerationRequest] = []

    def list_models(self) -> list[str]:
        return ["irodori-tts-mlx"]

    def status_metadata(self) -> dict[str, object]:
        return {"runtime": "compatibility", "configured": True, "loaded": True}

    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        self.requests.append(request)
        if request.irodori.get("num_steps") == 0:
            raise RuntimeRequestError("irodori.num_steps must be > 0.")
        return SpeechGenerationResult(audio=wav_bytes(), media_type="audio/wav")


def openai_client(
    runtime: CompatibilityRuntime,
    *,
    api_key: str = "local-dev-token",
    bearer_token: str | None = None,
) -> OpenAI:
    app = create_app(runtime=runtime, config=ServerConfig(bearer_token=bearer_token))
    return OpenAI(
        base_url="http://testserver/v1",
        api_key=api_key,
        http_client=TestClient(app),
    )


def test_openai_python_client_lists_models_with_bearer_auth() -> None:
    runtime = CompatibilityRuntime()
    client = openai_client(runtime, api_key="secret", bearer_token="secret")

    models = client.models.list()

    assert [model.id for model in models.data] == ["irodori-tts-mlx"]
    assert models.object == "list"


def test_openai_python_client_surfaces_bearer_auth_errors() -> None:
    runtime = CompatibilityRuntime()
    client = openai_client(runtime, api_key="wrong", bearer_token="secret")

    with pytest.raises(AuthenticationError) as exc_info:
        client.models.list()

    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.type == "authentication_error"


def test_openai_python_client_downloads_non_streaming_speech(tmp_path: Path) -> None:
    runtime = CompatibilityRuntime()
    client = openai_client(runtime)
    output_path = tmp_path / "speech.wav"

    response = client.audio.speech.create(
        model="irodori-tts-mlx",
        voice="voicedesign",
        input="Hello from the OpenAI Python client.",
        response_format="wav",
        extra_body={
            "irodori": {
                "no_ref": True,
                "caption": "calm narration, clear diction",
                "preset": "balanced",
            }
        },
    )
    response.write_to_file(output_path)

    assert response.content == wav_bytes()
    assert output_path.read_bytes() == wav_bytes()
    assert runtime.requests == [
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="Hello from the OpenAI Python client.",
            voice="voicedesign",
            response_format="wav",
            speed=1.0,
            irodori={
                "no_ref": True,
                "caption": "calm narration, clear diction",
                "preset": "balanced",
            },
        )
    ]


def test_openai_python_client_rejects_non_sse_streaming() -> None:
    runtime = CompatibilityRuntime()
    client = openai_client(runtime)

    with pytest.raises(BadRequestError) as exc_info:
        client.audio.speech.create(
            model="irodori-tts-mlx",
            voice="voicedesign",
            input="Only SSE speech streaming is supported.",
            response_format="wav",
            extra_body={"stream": True, "irodori": {"no_ref": True}},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "unsupported_streaming"
    assert exc_info.value.param == "stream"
    assert runtime.requests == []


def test_openai_python_client_receives_converted_pcm_response() -> None:
    runtime = CompatibilityRuntime()
    client = openai_client(runtime)

    response = client.audio.speech.create(
        model="irodori-tts-mlx",
        voice="voicedesign",
        input="Return raw PCM bytes.",
        response_format="pcm",
        extra_body={"irodori": {"no_ref": True}},
    )

    assert response.content == b"\x01\x02\x03\x04"
    assert runtime.requests[-1].response_format == "wav"


def test_supported_upstream_style_request_fixture_is_forwarded_to_runtime() -> None:
    runtime = CompatibilityRuntime()
    client = TestClient(create_app(runtime=runtime))
    upstream_style_payload = {
        "model": "irodori-tts-mlx",
        "input": "こんにちは。互換性テストです。",
        "voice": "voicedesign",
        "response_format": "wav",
        "speed": 1.2,
        "stream": False,
        "irodori": {
            "no_ref": True,
            "caption": "bright narration, clear diction",
            "preset": "fast",
            "seed": 42,
            "num_steps": 12,
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 3.5,
            "cfg_scale_speaker": 4.0,
            "chunking_enabled": True,
            "punctuation_chunking_enabled": True,
            "tail_trim_ms": 10,
            "tail_silence_trim_ms": 80,
            "tail_silence_keep_ms": 20,
            "tail_silence_threshold": 512,
        },
    }

    response = client.post("/v1/audio/speech", json=upstream_style_payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert runtime.requests == [
        SpeechGenerationRequest(
            model="irodori-tts-mlx",
            input="こんにちは。互換性テストです。",
            voice="voicedesign",
            response_format="wav",
            speed=1.2,
            irodori=upstream_style_payload["irodori"],
        )
    ]


@pytest.mark.parametrize(
    ("payload_patch", "status_code", "param", "code"),
    [
        (
            {"irodori": {"no_ref": True, "num_steps": 0}},
            400,
            "irodori",
            "invalid_irodori_options",
        ),
    ],
)
def test_intentionally_unsupported_upstream_style_fields_return_openai_errors(
    payload_patch: dict[str, object],
    status_code: int,
    param: str,
    code: str,
) -> None:
    runtime = CompatibilityRuntime()
    client = TestClient(create_app(runtime=runtime))
    payload = {
        "model": "irodori-tts-mlx",
        "input": "unsupported fixture",
        "voice": "voicedesign",
        "response_format": "wav",
    } | payload_patch

    response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == status_code
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == param
    assert response.json()["error"]["code"] == code

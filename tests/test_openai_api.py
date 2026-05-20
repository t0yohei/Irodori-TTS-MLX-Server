import asyncio
import base64
import errno
import importlib
import io
import json
import os
import threading
import time
import wave

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import irodori_tts_mlx_server.voices as voice_module
from irodori_tts_mlx_server import create_app
from irodori_tts_mlx_server.config import ServerConfig, server_config_from_env
from irodori_tts_mlx_server.factory import _install_voice_upload_size_guard
from irodori_tts_mlx_server.runtime import (
    MANAGED_REFERENCE_CACHE_OPTION,
    RuntimeRequestError,
    SpeechGenerationRequest,
    SpeechGenerationResult,
)
from irodori_tts_mlx_server.voices import VoiceRegistry


VOICE_FORMATS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm"]


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


class CacheTrackingSpeechRuntime(MockSpeechRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.configured_cache_entries: list[int] = []
        self.invalidated_voice_ids: list[str] = []

    def configure_managed_reference_cache(self, *, max_entries: int) -> None:
        self.configured_cache_entries.append(max_entries)

    def invalidate_managed_reference_cache(self, voice_id: str) -> None:
        self.invalidated_voice_ids.append(voice_id)


class FalseyMockSpeechRuntime(MockSpeechRuntime):
    def __len__(self) -> int:
        return 0


class InvalidOptionsRuntime(MockSpeechRuntime):
    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeRequestError("irodori.num_steps must be > 0.")


class UnsupportedLoraRuntime(MockSpeechRuntime):
    def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
        raise RuntimeRequestError(
            "irodori.lora_adapter is not supported by the current Irodori-TTS-MLX runtime boundary."
        )


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


def assert_managed_reference_options(
    options: dict[str, object],
    *,
    path: str,
    voice_id: str,
) -> None:
    assert options["ref_wav"] == path
    assert options["no_ref"] is False
    cache = options[MANAGED_REFERENCE_CACHE_OPTION]
    assert isinstance(cache, dict)
    assert cache["voice_id"] == voice_id
    assert cache["path"] == path
    assert cache["size"] >= 0
    assert cache["mtime_ns"] > 0


def sse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


def speech_audio_done_payload() -> dict[str, object]:
    return {
        "type": "speech.audio.done",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


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
    upload_response = client.post(
        "/v1/audio/voices",
        content=b"not multipart",
        headers={"content-type": "multipart/form-data; boundary=bad"},
    )
    replace_response = client.put(
        "/v1/audio/voices/sample",
        content=b"not multipart",
        headers={"content-type": "multipart/form-data; boundary=bad"},
    )
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
    assert upload_response.status_code == 401
    assert replace_response.status_code == 401
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
    assert_managed_reference_options(
        runtime.requests[-1].irodori,
        path=str(tmp_path / "voices" / "sample.wav"),
        voice_id="sample",
    )
    assert replaced.status_code == 200
    assert replaced.json()["bytes"] == len(b"new wav")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": "sample", "object": "voice_file", "deleted": True}
    assert missing.status_code == 404


def test_voice_cache_limit_configures_runtime_and_mutations_invalidate_voice(tmp_path) -> None:
    runtime = CacheTrackingSpeechRuntime()
    client = TestClient(
        create_app(
            runtime=runtime,
            config=ServerConfig(
                voices_dir=tmp_path / "voices",
                reference_cache_max_entries=2,
            ),
        )
    )

    created = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"old wav", "audio/wav")},
        data={"voice_id": "sample"},
    )
    replaced = client.put(
        "/v1/audio/voices/sample",
        files={"file": ("sample.wav", b"new wav", "audio/wav")},
    )
    deleted = client.delete("/v1/audio/voices/sample")

    assert created.status_code == 201
    assert replaced.status_code == 200
    assert deleted.status_code == 200
    assert runtime.configured_cache_entries == [2]
    assert runtime.invalidated_voice_ids == ["sample", "sample", "sample"]


def test_voice_upload_accepts_managed_non_wav_and_voice_object_resolution(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(
        create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path / "voices"))
    )

    created = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.flac", b"flac", "audio/flac")},
        data={"voice_id": "sample"},
    )
    listed = client.get("/v1/audio/voices")
    speech = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": {"id": "sample"},
            "response_format": "wav",
        },
    )

    assert created.status_code == 201
    assert created.json()["filename"] == "sample.flac"
    assert listed.status_code == 200
    assert listed.json()["data"][0]["ref_wav"] == str(tmp_path / "voices" / "sample.flac")
    assert speech.status_code == 200
    assert runtime.requests[-1].voice == "sample"
    assert_managed_reference_options(
        runtime.requests[-1].irodori,
        path=str(tmp_path / "voices" / "sample.flac"),
        voice_id="sample",
    )


@pytest.mark.parametrize("voice", [{"id": False}, {"id": {"nested": "sample"}}, {"id": ""}, False])
def test_audio_speech_rejects_non_string_voice_object_ids(tmp_path, voice) -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path)))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": voice,
            "response_format": "wav",
        },
    )

    assert response.status_code == 422
    assert runtime.requests == []


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
        files={"file": ("sample.txt", b"text", "text/plain")},
        data={"voice_id": "text"},
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
    assert "Managed reference voices must use one of:" in bad_extension.json()["error"]["message"]
    assert empty.status_code == 400
    assert empty.json()["error"]["message"] == "Voice file must not be empty."
    assert not (tmp_path.parent / "sample.wav").exists()


def test_voice_upload_reports_storage_errors(monkeypatch, tmp_path) -> None:
    def fail_write_file(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(VoiceRegistry, "write_file", fail_write_file)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    response = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "voice_storage_unavailable"


def test_voice_create_removes_partial_file_after_atomic_link_failure(monkeypatch, tmp_path) -> None:
    def fail_link(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(voice_module.os, "link", fail_link)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    response = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "voice_storage_unavailable"
    assert not (tmp_path / "sample.wav").exists()
    assert list(tmp_path.iterdir()) == []


def test_voice_create_falls_back_when_hard_links_are_unsupported(monkeypatch, tmp_path) -> None:
    def fail_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "operation not supported")

    monkeypatch.setattr(voice_module.os, "link", fail_link)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    response = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )

    assert response.status_code == 201
    assert (tmp_path / "sample.wav").read_bytes() == b"wav"


def test_voice_create_removes_fallback_file_when_close_fails(monkeypatch, tmp_path) -> None:
    def fail_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "operation not supported")

    original_fdopen = voice_module.os.fdopen

    class FailingCloseFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *exc_info):
            self.file.__exit__(*exc_info)
            raise OSError(errno.ENOSPC, "no space left")

        def write(self, data):
            return self.file.write(data)

    def fail_fdopen(*args, **kwargs):
        return FailingCloseFile(original_fdopen(*args, **kwargs))

    monkeypatch.setattr(voice_module.os, "link", fail_link)
    monkeypatch.setattr(voice_module.os, "fdopen", fail_fdopen)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    response = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "voice_storage_unavailable"
    assert not (tmp_path / "sample.wav").exists()

    monkeypatch.setattr(voice_module.os, "fdopen", original_fdopen)
    retry_response = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )

    assert retry_response.status_code == 201
    assert (tmp_path / "sample.wav").read_bytes() == b"wav"


def test_voice_delete_reports_storage_errors(monkeypatch, tmp_path) -> None:
    def fail_delete_file(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(VoiceRegistry, "delete_file", fail_delete_file)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    response = client.delete("/v1/audio/voices/sample")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "voice_storage_unavailable"


def test_voice_replace_reports_storage_race_as_storage_error(monkeypatch, tmp_path) -> None:
    (tmp_path / "sample.wav").write_bytes(b"old wav")

    def fail_write_file(*args, **kwargs):
        raise FileNotFoundError("voices directory disappeared")

    monkeypatch.setattr(VoiceRegistry, "write_file", fail_write_file)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    response = client.put(
        "/v1/audio/voices/sample",
        files={"file": ("sample.wav", b"new wav", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "voice_storage_unavailable"


def test_voice_upload_reports_file_storage_root_as_storage_error(tmp_path) -> None:
    voices_root = tmp_path / "voices"
    voices_root.write_text("not a directory")
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=voices_root))
    )

    response = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"wav", "audio/wav")},
        data={"voice_id": "sample"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "voice_storage_unavailable"


def test_voice_read_routes_report_file_storage_root_as_storage_error_without_breaking_speech(
    tmp_path,
) -> None:
    voices_root = tmp_path / "voices"
    voices_root.write_text("not a directory")
    runtime = MockSpeechRuntime()
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=voices_root)))

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
    health = client.get("/health")

    assert listed.status_code == 503
    assert listed.json()["error"]["code"] == "voice_storage_unavailable"
    assert fetched.status_code == 503
    assert fetched.json()["error"]["code"] == "voice_storage_unavailable"
    assert speech.status_code == 200
    assert runtime.requests[-1].irodori == {}
    assert health.status_code == 200
    assert "not a directory" in health.json()["server"]["voices"]["files_error"]


def test_voice_list_reports_storage_errors_without_breaking_health(monkeypatch, tmp_path) -> None:
    def fail_list_files(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(VoiceRegistry, "list_files", fail_list_files)
    client = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=tmp_path))
    )

    health = client.get("/health")
    response = client.get("/v1/audio/voices")

    assert health.status_code == 200
    assert health.json()["server"]["voices"]["files"] == 0
    assert "permission denied" in health.json()["server"]["voices"]["files_error"]
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "voice_storage_unavailable"


def test_voice_upload_and_replace_reject_files_above_configured_limit(tmp_path) -> None:
    client = TestClient(
        create_app(
            runtime=MockSpeechRuntime(),
            config=ServerConfig(voices_dir=tmp_path, max_voice_upload_bytes=3),
        )
    )
    assert (
        client.post(
            "/v1/audio/voices",
            files={"file": ("sample.wav", b"wav", "audio/wav")},
            data={"voice_id": "sample"},
        ).status_code
        == 201
    )

    too_large = client.post(
        "/v1/audio/voices",
        files={"file": ("large.wav", b"1234", "audio/wav")},
        data={"voice_id": "large"},
    )
    replace_too_large = client.put(
        "/v1/audio/voices/sample",
        files={"file": ("sample.wav", b"1234", "audio/wav")},
    )

    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "voice_file_too_large"
    assert replace_too_large.status_code == 413
    assert replace_too_large.json()["error"]["code"] == "voice_file_too_large"
    assert (tmp_path / "sample.wav").read_bytes() == b"wav"
    assert not (tmp_path / "large.wav").exists()


@pytest.mark.parametrize(
    "ref_wav",
    [
        "https://example.com/ref.wav",
        "../outside.wav",
        "/tmp/ref.wav",
        "imports/private.wav",
        "bad.id.wav",
    ],
)
def test_audio_speech_rejects_remote_or_arbitrary_reference_wav(tmp_path, ref_wav) -> None:
    runtime = MockSpeechRuntime()
    (tmp_path / "imports").mkdir()
    (tmp_path / "imports" / "private.wav").write_bytes(b"wav")
    (tmp_path / "bad.id.wav").write_bytes(b"wav")
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path)))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "sample",
            "response_format": "wav",
            "irodori": {"ref_wav": ref_wav, "no_ref": False},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "irodori.ref_wav"
    assert response.json()["error"]["code"] == "invalid_irodori_options"
    assert runtime.requests == []


def test_audio_speech_accepts_explicit_managed_reference_wav(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    (tmp_path / "sample.wav").write_bytes(b"wav")
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path)))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "sample",
            "response_format": "wav",
            "irodori": {"ref_wav": "sample.wav", "no_ref": False},
        },
    )

    assert response.status_code == 200
    assert_managed_reference_options(
        runtime.requests[-1].irodori,
        path=str(tmp_path / "sample.wav"),
        voice_id="sample",
    )


def test_voice_upload_rejects_large_content_length_before_multipart_spooling(tmp_path) -> None:
    client = TestClient(
        create_app(
            runtime=MockSpeechRuntime(),
            config=ServerConfig(voices_dir=tmp_path, max_voice_upload_bytes=3),
        )
    )

    response = client.post(
        "/v1/audio/voices",
        files={"file": ("large.wav", b"x" * 70000, "audio/wav")},
        data={"voice_id": "large"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "voice_file_too_large"
    assert not (tmp_path / "large.wav").exists()


def test_voice_upload_without_content_length_installs_streamed_request_limit() -> None:
    messages = [
        {"type": "http.request", "body": b"1" * 70000, "more_body": False},
    ]

    async def receive():
        return messages.pop(0)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/voices",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "client": ("testclient", 50000),
        },
        receive,
    )

    _install_voice_upload_size_guard(
        ServerConfig(max_voice_upload_bytes=1),
        request,
    )

    async def read_stream() -> None:
        async for _chunk in request.stream():
            pass

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(read_stream())

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"]["code"] == "voice_file_too_large"


def test_voice_list_ignores_wav_files_with_unmanaged_ids(tmp_path) -> None:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "sample.wav").write_bytes(b"wav")
    (voices_dir / "speaker.v1.wav").write_bytes(b"wav")
    (voices_dir / "upper.WAV").write_bytes(b"wav")

    response = TestClient(
        create_app(runtime=MockSpeechRuntime(), config=ServerConfig(voices_dir=voices_dir))
    ).get("/v1/audio/voices")

    assert response.status_code == 200
    assert [voice["id"] for voice in response.json()["data"]] == ["sample"]


def test_voice_resolution_ignores_symlinked_wav_files(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    outside_target = tmp_path / "outside.wav"
    outside_target.write_bytes(b"outside")
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "sample.wav").symlink_to(outside_target)
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=voices_dir)))

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

    assert listed.status_code == 200
    assert listed.json()["data"] == []
    assert fetched.status_code == 404
    assert speech.status_code == 200
    assert runtime.requests[-1].irodori == {}


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
            "irodori": {"no_ref": True, "caption": "calm"},
        },
    )

    assert response.status_code == 200
    assert runtime.requests[-1].irodori == {"no_ref": True, "caption": "calm"}


def test_managed_voice_resolution_accepts_upstream_no_ref_false_alias(tmp_path) -> None:
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
            "no_ref": False,
        },
    )

    assert response.status_code == 200
    assert_managed_reference_options(
        runtime.requests[-1].irodori,
        path=str(tmp_path / "sample.wav"),
        voice_id="sample",
    )


def test_short_managed_voice_fast_preset_auto_selects_conservative_seconds(tmp_path) -> None:
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
            "input": "もう一度試しています。",
            "voice": "sample",
            "response_format": "wav",
            "irodori": {"preset": "fast"},
        },
    )

    assert response.status_code == 200
    options = runtime.requests[-1].irodori
    assert options["seconds"] == 1.68
    assert options["preset"] == "fast"
    assert_managed_reference_options(
        options,
        path=str(tmp_path / "sample.wav"),
        voice_id="sample",
    )


def test_managed_voice_auto_seconds_preserves_explicit_seconds(tmp_path) -> None:
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
            "input": "もう一度試しています。",
            "voice": "sample",
            "response_format": "wav",
            "seconds": 2.4,
            "irodori": {"preset": "fast"},
        },
    )

    assert response.status_code == 200
    assert runtime.requests[-1].irodori["seconds"] == 2.4


@pytest.mark.parametrize(
    "input_text,irodori",
    [
        ("これは短文ですが品質プリセットなので自動秒数を使いません。", {"preset": "balanced"}),
        (
            "長めの文章では過度な短縮で末尾が切れる可能性があるため自動秒数を使いません。" * 3,
            {"preset": "fast"},
        ),
        (
            "速度指定がある場合はduration_scaleとの二重制御を避けます。",
            {"preset": "fast", "duration_scale": 0.9},
        ),
    ],
)
def test_managed_voice_auto_seconds_gating_noops(tmp_path, input_text, irodori) -> None:
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
            "input": input_text,
            "voice": "sample",
            "response_format": "wav",
            "irodori": irodori,
        },
    )

    assert response.status_code == 200
    assert "seconds" not in runtime.requests[-1].irodori


@pytest.mark.parametrize(
    "payload_patch",
    [
        {"irodori": {"no_ref": "true", "caption": "calm"}},
        {"no_ref": "true", "irodori": {"caption": "calm"}},
    ],
)
def test_managed_voice_resolution_respects_boolean_like_no_ref_true(
    tmp_path, payload_patch
) -> None:
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
        }
        | payload_patch,
    )

    assert response.status_code == 200
    assert runtime.requests[-1].irodori == {"no_ref": "true", "caption": "calm"}


@pytest.mark.parametrize("voice", ["../sample", "my.voice", "voice:v1"])
def test_audio_speech_treats_unmanaged_punctuated_voice_as_reference_miss(tmp_path, voice) -> None:
    runtime = MockSpeechRuntime()
    client = TestClient(create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path)))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": voice,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert runtime.requests[-1].voice == voice
    assert runtime.requests[-1].irodori == {}


def test_voice_replace_rejects_symlink_without_overwriting_target(tmp_path) -> None:
    outside_target = tmp_path / "outside.wav"
    outside_target.write_bytes(b"outside")
    voice_path = tmp_path / "sample.wav"
    voice_path.symlink_to(outside_target)

    with pytest.raises(ValueError, match="symbolic links"):
        VoiceRegistry(tmp_path).write_file(
            voice_id="sample",
            filename="sample.wav",
            data=b"replacement",
            replace=True,
        )

    assert outside_target.read_bytes() == b"outside"


def test_voice_replace_ignores_same_id_extension_directory(tmp_path) -> None:
    (tmp_path / "sample.wav").write_bytes(b"old wav")
    (tmp_path / "sample.mp3").mkdir()

    voice_file = VoiceRegistry(tmp_path).write_file(
        voice_id="sample",
        filename="sample.flac",
        data=b"new flac",
        replace=True,
    )

    assert voice_file.path == tmp_path / "sample.flac"
    assert (tmp_path / "sample.flac").read_bytes() == b"new flac"
    assert not (tmp_path / "sample.wav").exists()
    assert (tmp_path / "sample.mp3").is_dir()


def test_voice_create_uses_atomic_exclusive_file_creation(tmp_path) -> None:
    registry = VoiceRegistry(tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def create_voice() -> None:
        barrier.wait(timeout=2)
        try:
            registry.write_file(
                voice_id="sample",
                filename="sample.wav",
                data=b"wav",
                replace=False,
            )
        except FileExistsError:
            outcome = "exists"
        else:
            outcome = "created"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=create_voice) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(outcomes) == ["created", "exists"]
    assert (tmp_path / "sample.wav").read_bytes() == b"wav"


def test_audio_speech_logs_request_lifecycle_without_request_text(caplog) -> None:
    caplog.set_level("INFO", logger="irodori_tts_mlx_server.server")
    runtime = MockSpeechRuntime()
    client = TestClient(create_app(runtime=runtime))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "private text should not be logged",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records]
    assert any("request_start method=POST path=/v1/audio/speech" in message for message in messages)
    assert any(
        "request_end method=POST path=/v1/audio/speech status_code=200" in message
        for message in messages
    )
    assert "private text should not be logged" not in caplog.text


def test_audio_speech_logs_request_end_when_unexpected_error_raises(caplog) -> None:
    class CrashingRuntime(MockSpeechRuntime):
        def generate_speech(self, request: SpeechGenerationRequest) -> SpeechGenerationResult:
            raise RuntimeError("boom")

    caplog.set_level("INFO", logger="irodori_tts_mlx_server.server")
    client = TestClient(create_app(runtime=CrashingRuntime()), raise_server_exceptions=False)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "private text should not be logged",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 500
    messages = [record.getMessage() for record in caplog.records]
    assert any("request_start method=POST path=/v1/audio/speech" in message for message in messages)
    assert any(
        "request_end method=POST path=/v1/audio/speech status_code=500" in message
        for message in messages
    )
    assert "private text should not be logged" not in caplog.text


def test_voice_create_keeps_voice_id_unique_across_extensions(tmp_path) -> None:
    registry = VoiceRegistry(tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def create_voice(filename: str, data: bytes) -> None:
        barrier.wait(timeout=2)
        try:
            registry.write_file(
                voice_id="sample",
                filename=filename,
                data=data,
                replace=False,
            )
        except FileExistsError:
            outcome = "exists"
        else:
            outcome = filename
        with lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=create_voice, args=("sample.wav", b"wav")),
        threading.Thread(target=create_voice, args=("sample.flac", b"flac")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(outcomes) in (["exists", "sample.flac"], ["exists", "sample.wav"])
    assert len(list(tmp_path.glob("sample.*"))) == 1
    assert not (tmp_path / ".sample.create.lock").exists()


def test_voice_create_recovers_stale_lock_when_voice_file_is_absent(tmp_path) -> None:
    lock_path = tmp_path / ".sample.create.lock"
    lock_path.write_bytes(b"")
    stale_time = time.time() - voice_module.STALE_CREATE_LOCK_SECONDS - 1
    os.utime(lock_path, (stale_time, stale_time))

    voice_file = VoiceRegistry(tmp_path).write_file(
        voice_id="sample",
        filename="sample.flac",
        data=b"flac",
        replace=False,
    )

    assert voice_file.path == tmp_path / "sample.flac"
    assert voice_file.path.read_bytes() == b"flac"
    assert not lock_path.exists()


def test_voice_list_and_delete_handle_preexisting_duplicate_extensions(tmp_path) -> None:
    (tmp_path / "sample.wav").write_bytes(b"wav")
    (tmp_path / "sample.flac").write_bytes(b"flac")

    registry = VoiceRegistry(tmp_path)
    listed = registry.list_files()
    deleted = registry.delete_file("sample")

    assert [voice.voice_id for voice in listed] == ["sample"]
    assert listed[0].path == tmp_path / "sample.wav"
    assert deleted is True
    assert not (tmp_path / "sample.wav").exists()
    assert not (tmp_path / "sample.flac").exists()


def test_audio_speech_times_out_when_synthesis_queue_is_full(caplog) -> None:
    caplog.set_level("WARNING", logger="irodori_tts_mlx_server.server")
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
    assert "synthesis_queue_timeout queue_timeout_seconds=0.01" in caplog.text


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


def test_audio_speech_sse_stream_returns_queue_timeout_event() -> None:
    runtime = BlockingSpeechRuntime()
    client = TestClient(
        create_app(
            runtime=runtime,
            config=ServerConfig(max_concurrent_synthesis=1, queue_timeout_seconds=0.01),
        )
    )
    payload = {
        "model": "irodori-tts-mlx",
        "input": "hello. goodbye.",
        "voice": "voicedesign",
        "response_format": "wav",
        "stream_format": "sse",
        "irodori": {"no_ref": True},
    }
    first_status: dict[str, int] = {}

    def first_request() -> None:
        first_status["status_code"] = client.post(
            "/v1/audio/speech",
            headers={"accept": "text/event-stream"},
            json=payload,
        ).status_code

    thread = threading.Thread(target=first_request)
    thread.start()
    assert runtime.started.wait(timeout=2)
    try:
        response = client.post(
            "/v1/audio/speech",
            headers={"accept": "text/event-stream"},
            json=payload,
        )
    finally:
        runtime.release.set()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert sse_events(response.text) == [
        (
            "error",
            {
                "error": {
                    "message": "Synthesis queue is full or the model is still loading; retry later.",
                    "type": "server_error",
                    "param": None,
                    "code": "synthesis_queue_timeout",
                }
            },
        )
    ]
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
                "no_ref": True,
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
                "no_ref": True,
                "preset": "balanced",
                "cfg_scale_caption": 3.5,
                "seconds": 2.0,
                "seed": 1234,
            },
        )
    ]


def test_audio_speech_accepts_upstream_style_top_level_option_aliases() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "no_ref": True,
            "seconds": 2.5,
            "duration_scale": 0.9,
            "num_steps": 24,
            "seed": 7,
            "cfg_scale_text": 3.1,
            "cfg_scale_caption": 3.2,
            "cfg_scale_speaker": 5.1,
            "cfg_guidance_mode": "independent",
            "cfg_min_t": 0.2,
            "cfg_max_t": 0.8,
            "t_schedule_mode": "sway",
            "sway_coeff": -0.75,
            "rescale_k": 0.7,
            "rescale_sigma": 1.2,
            "speaker_kv_scale": 1.4,
            "speaker_kv_min_t": 0.8,
            "speaker_kv_max_layers": 3,
            "max_ref_seconds": 12.0,
            "context_kv_cache": False,
            "chunking_enabled": False,
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori == {
        "no_ref": True,
        "seconds": 2.5,
        "duration_scale": 0.9,
        "num_steps": 24,
        "seed": 7,
        "cfg_scale_text": 3.1,
        "cfg_scale_caption": 3.2,
        "cfg_scale_speaker": 5.1,
        "cfg_guidance_mode": "independent",
        "cfg_min_t": 0.2,
        "cfg_max_t": 0.8,
        "t_schedule_mode": "sway",
        "sway_coeff": -0.75,
        "rescale_k": 0.7,
        "rescale_sigma": 1.2,
        "speaker_kv_scale": 1.4,
        "speaker_kv_min_t": 0.8,
        "speaker_kv_max_layers": 3,
        "max_ref_seconds": 12.0,
        "context_kv_cache": False,
        "chunking_enabled": False,
    }


def test_audio_speech_accepts_upstream_style_irodori_option_aliases(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    (tmp_path / "reference.wav").write_bytes(b"wav")
    response = TestClient(
        create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path))
    ).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {
                "ref_wav": "reference.wav",
                "no_ref": False,
                "max_ref_seconds": 10,
                "context_kv_cache": True,
                "chunking_enabled": False,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori["max_ref_seconds"] == 10
    assert runtime.requests[0].irodori["context_kv_cache"] is True
    assert runtime.requests[0].irodori["chunking_enabled"] is False
    assert_managed_reference_options(
        runtime.requests[0].irodori,
        path=str(tmp_path / "reference.wav"),
        voice_id="reference",
    )


def test_audio_speech_accepts_ref_embed_without_managed_voice_injection(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    (tmp_path / "sample.wav").write_bytes(b"wav")
    ref_embed = tmp_path / "sample.speaker.safetensors"
    ref_embed.write_bytes(b"speaker")
    response = TestClient(
        create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path))
    ).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "sample",
            "response_format": "wav",
            "irodori": {
                "ref_embed": "sample.speaker.safetensors",
                "t_schedule_mode": "sway",
                "sway_coeff": -0.75,
                "rescale_k": 0.7,
                "rescale_sigma": 1.2,
                "speaker_kv_scale": 1.4,
                "speaker_kv_min_t": 0.8,
                "speaker_kv_max_layers": 3,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori == {
        "ref_embed": str(ref_embed),
        "t_schedule_mode": "sway",
        "sway_coeff": -0.75,
        "rescale_k": 0.7,
        "rescale_sigma": 1.2,
        "speaker_kv_scale": 1.4,
        "speaker_kv_min_t": 0.8,
        "speaker_kv_max_layers": 3,
    }


def test_audio_speech_accepts_top_level_ref_embed_alias(tmp_path) -> None:
    runtime = MockSpeechRuntime()
    ref_embed = tmp_path / "sample.speaker.safetensors"
    ref_embed.write_bytes(b"speaker")
    response = TestClient(
        create_app(runtime=runtime, config=ServerConfig(voices_dir=tmp_path))
    ).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "ref_embed": "sample.speaker.safetensors",
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori == {
        "ref_embed": str(ref_embed),
    }


def test_audio_speech_rejects_arbitrary_ref_embed_path() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {"ref_embed": "/tmp/sample.speaker.safetensors"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "irodori.ref_embed"
    assert "must resolve inside" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("payload_patch", "expected_irodori"),
    [
        (
            {"no_ref": False, "irodori": {"no_ref": "false"}},
            {"no_ref": False},
        ),
        (
            {"chunking_enabled": False, "irodori": {"chunking_enabled": "false"}},
            {"chunking_enabled": False},
        ),
    ],
)
def test_audio_speech_accepts_semantically_equal_bool_alias_values(
    payload_patch, expected_irodori
) -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
        }
        | payload_patch,
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori == expected_irodori


@pytest.mark.parametrize(
    "unsupported",
    ["ref_latent", "reference_wav", "no_reference", "max_reference_seconds", "no_context_kv_cache"],
)
def test_audio_speech_rejects_unsupported_upstream_irodori_options(unsupported) -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {unsupported: "value"},
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == "validation_error"
    assert unsupported in response.json()["error"]["message"]


def test_audio_speech_rejects_conflicting_alias_values() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "no_ref": True,
            "irodori": {"no_ref": False},
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "conflicts with irodori.no_ref" in response.json()["error"]["message"]


def test_audio_speech_rejects_unmanaged_top_level_ref_wav() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "voicedesign",
            "response_format": "wav",
            "ref_wav": "/tmp/reference.wav",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "irodori.ref_wav"
    assert response.json()["error"]["code"] == "invalid_irodori_options"


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
                "no_ref": True,
                "chunking_enabled": True,
                "tail_trim_ms": 20,
                "tail_silence_trim_ms": 120,
                "tail_silence_keep_ms": 40,
                "tail_silence_threshold": 512,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].irodori == {
        "no_ref": True,
        "chunking_enabled": True,
        "tail_trim_ms": 20,
        "tail_silence_trim_ms": 120,
        "tail_silence_keep_ms": 40,
        "tail_silence_threshold": 512,
    }


def test_audio_speech_rejects_unsupported_chunking_controls() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello. goodbye.",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {
                "no_ref": True,
                "chunking": True,
                "chunk_max_chars": 8,
            },
        },
    )

    assert response.status_code == 422
    detail = response.json()["error"]["message"]
    assert "irodori.chunk_max_chars" in detail
    assert "irodori.chunking" in detail
    assert runtime.requests == []


def test_audio_speech_sse_stream_emits_each_generated_chunk_as_audio_delta() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        headers={"accept": "text/event-stream"},
        json={
            "model": "irodori-tts-mlx",
            "input": "こんにちは。さようなら。",
            "voice": "voicedesign",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "no_ref": True,
                "chunking_enabled": True,
                "punctuation_chunking_enabled": True,
                "chunk_min_chars": 1,
                "seconds": 3.0,
            },
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [request.input for request in runtime.requests] == ["こんにちは。", "さようなら。"]
    assert [request.irodori["chunking_enabled"] for request in runtime.requests] == [False, False]
    assert [round(request.irodori["seconds"], 4) for request in runtime.requests] == [
        1.5,
        1.5,
    ]
    events = sse_events(response.text)
    assert [event for event, _data in events] == [
        "speech.audio.delta",
        "speech.audio.delta",
        "speech.audio.done",
    ]
    first_chunk = events[0][1]
    assert first_chunk["type"] == "speech.audio.delta"
    assert base64.b64decode(first_chunk["audio"]) == wav_bytes()
    assert events[2][1] == speech_audio_done_payload()


def test_audio_speech_sse_stream_supports_punctuation_chunking_enabled() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        headers={"accept": "text/event-stream"},
        json={
            "model": "irodori-tts-mlx",
            "input": (
                "こんにちは。これは stream chunks の動作確認です。"
                "最初の音声が返ったら、続きの音声を生成しながら再生できます。"
            ),
            "voice": "voicedesign",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "no_ref": True,
                "chunking_enabled": True,
                "punctuation_chunking_enabled": True,
            },
        },
    )

    assert response.status_code == 200
    assert [request.input for request in runtime.requests] == [
        (
            "こんにちは。これは stream chunks の動作確認です。"
            "最初の音声が返ったら、続きの音声を生成しながら再生できます。"
        ),
    ]
    events = sse_events(response.text)
    assert [event for event, _data in events] == [
        "speech.audio.delta",
        "speech.audio.done",
    ]
    assert events[0][1]["type"] == "speech.audio.delta"
    assert events[1][1] == speech_audio_done_payload()


def test_audio_speech_sse_stream_supports_first_sentence_comma_chunking() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        headers={"accept": "text/event-stream"},
        json={
            "model": "irodori-tts-mlx",
            "input": "最初は速く、すぐ返します。次は長くて、通常のままです。",
            "voice": "voicedesign",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "no_ref": True,
                "chunking_enabled": True,
                "punctuation_chunking_enabled": True,
                "first_sentence_comma_chunking_enabled": True,
            },
        },
    )

    assert response.status_code == 200
    assert [request.input for request in runtime.requests] == [
        "最初は速く、",
        "すぐ返します。",
        "次は長くて、通常のままです。",
    ]
    events = sse_events(response.text)
    assert [event for event, _data in events] == [
        "speech.audio.delta",
        "speech.audio.delta",
        "speech.audio.delta",
        "speech.audio.done",
    ]
    assert [data["type"] for _event, data in events[:3]] == [
        "speech.audio.delta",
        "speech.audio.delta",
        "speech.audio.delta",
    ]


def test_audio_speech_sse_stream_rejects_invalid_first_sentence_comma_chunking() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        headers={"accept": "text/event-stream"},
        json={
            "model": "irodori-tts-mlx",
            "input": "こんにちは、テストです。",
            "voice": "voicedesign",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "no_ref": True,
                "punctuation_chunking_enabled": True,
                "first_sentence_comma_chunking_enabled": "quick",
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "irodori.first_sentence_comma_chunking_enabled"
    assert error["code"] == "invalid_irodori_options"
    assert runtime.requests == []


def test_audio_speech_sse_stream_rejects_unsupported_punctuation_chunk_mode() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        headers={"accept": "text/event-stream"},
        json={
            "model": "irodori-tts-mlx",
            "input": "こんにちは。互換性テストです。",
            "voice": "voicedesign",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "no_ref": True,
                "chunking_enabled": True,
                "chunk_mode": "punctuation",
            },
        },
    )

    assert response.status_code == 422
    assert "Unsupported upstream Irodori option(s): irodori.chunk_mode." in str(
        response.json()["error"]["message"]
    )
    assert runtime.requests == []


def test_audio_speech_rejects_unsupported_chunk_size_options() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "こんにちは。互換性テストです。",
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {
                "no_ref": True,
                "chunking_enabled": True,
                "punctuation_chunking_enabled": True,
                "chunk_hard_max_chars": 3,
            },
        },
    )

    assert response.status_code == 422
    assert "Unsupported upstream Irodori option(s): irodori.chunk_hard_max_chars." in str(
        response.json()["error"]["message"]
    )
    assert runtime.requests == []


def test_audio_speech_sse_stream_respects_disabled_chunking() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello. goodbye.",
            "voice": "voicedesign",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "no_ref": True,
                "chunking_enabled": False,
            },
        },
    )

    assert response.status_code == 200
    assert [request.input for request in runtime.requests] == ["hello. goodbye."]
    events = sse_events(response.text)
    assert [event for event, _data in events] == [
        "speech.audio.delta",
        "speech.audio.done",
    ]
    assert events[0][1]["type"] == "speech.audio.delta"
    assert events[1][1] == speech_audio_done_payload()


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


def test_audio_speech_rejects_non_sse_stream_true() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
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
        "message": "Only stream_format='sse' is supported for speech streaming.",
        "type": "invalid_request_error",
        "param": "stream",
        "code": "unsupported_streaming",
    }
    assert runtime.requests == []


def test_audio_speech_rejects_audio_stream_format() -> None:
    runtime = MockSpeechRuntime()
    response = TestClient(create_app(runtime=runtime)).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "stream_format": "audio",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["message"] == "Input should be 'sse'"
    assert runtime.requests == []


def test_audio_speech_accepts_sse_accept_header() -> None:
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

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event for event, _data in sse_events(response.text)] == [
        "speech.audio.delta",
        "speech.audio.done",
    ]


def test_audio_speech_rejects_unknown_stream_format() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "stream_format": "events",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["message"] == "Input should be 'sse'"


def test_audio_speech_stream_chunks_route_is_removed() -> None:
    response = TestClient(create_app(runtime=MockSpeechRuntime())).post(
        "/v1/audio/speech/stream-chunks",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        },
    )

    assert response.status_code == 404


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


def test_audio_speech_rejects_unsupported_lora_adapter_with_openai_error_shape() -> None:
    response = TestClient(create_app(runtime=UnsupportedLoraRuntime())).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts-mlx",
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "irodori": {"lora_adapter": "warm-narration"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "irodori.lora_adapter is not supported by the current Irodori-TTS-MLX runtime boundary.",
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
    monkeypatch.setenv("IRODORI_MLX_MAX_TEXT_LEN", "not-an-int")

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
        "load_state": "failed",
        "model_id": "irodori-tts-mlx",
        "last_load_error": "IRODORI_MLX_MAX_TEXT_LEN must be an integer.",
    }
    assert health_response.json()["server"]["auth_enabled"] is False
    assert speech_response.status_code == 503
    assert speech_response.json()["error"]["code"] == "runtime_unavailable"
    assert "IRODORI_MLX_MAX_TEXT_LEN" in speech_response.json()["error"]["message"]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_server_config_rejects_non_finite_queue_timeout_env(monkeypatch, value: str) -> None:
    monkeypatch.setenv("IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS"):
        server_config_from_env()


def test_server_config_rejects_non_positive_voice_upload_limit_env(monkeypatch) -> None:
    monkeypatch.setenv("IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES", "0")

    with pytest.raises(ValueError, match="IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES"):
        server_config_from_env()


def test_server_config_reads_reference_cache_limit_env(monkeypatch) -> None:
    monkeypatch.setenv("IRODORI_SERVER_REFERENCE_CACHE_MAX_ENTRIES", "0")

    assert server_config_from_env().reference_cache_max_entries == 0


def test_server_config_rejects_negative_reference_cache_limit_env(monkeypatch) -> None:
    monkeypatch.setenv("IRODORI_SERVER_REFERENCE_CACHE_MAX_ENTRIES", "-1")

    with pytest.raises(ValueError, match="IRODORI_SERVER_REFERENCE_CACHE_MAX_ENTRIES"):
        server_config_from_env()


def test_invalid_server_env_keeps_health_available_and_blocks_openai_routes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
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
            "formats": VOICE_FORMATS,
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

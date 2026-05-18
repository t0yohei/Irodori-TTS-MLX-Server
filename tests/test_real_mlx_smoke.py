import os
from io import BytesIO
import wave

import pytest
from fastapi.testclient import TestClient

from irodori_tts_mlx_server.config import ServerConfig
from irodori_tts_mlx_server.factory import create_app


SMOKE_ENABLED_ENV = "IRODORI_REAL_MLX_SMOKE"
WEIGHT_SOURCE_ENVS = (
    "IRODORI_MLX_WEIGHTS_REPO",
    "IRODORI_MLX_WEIGHTS_DIR",
)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_weight_sources() -> list[str]:
    return [name for name in WEIGHT_SOURCE_ENVS if os.getenv(name)]


pytestmark = [
    pytest.mark.real_mlx,
    pytest.mark.skipif(
        not _truthy_env(SMOKE_ENABLED_ENV),
        reason=f"set {SMOKE_ENABLED_ENV}=1 to run the real MLX smoke test",
    ),
    pytest.mark.skipif(
        not _configured_weight_sources(),
        reason=("set IRODORI_MLX_WEIGHTS_REPO or IRODORI_MLX_WEIGHTS_DIR"),
    ),
]


def test_real_mlx_voicedesign_speech_endpoint_returns_non_empty_wav() -> None:
    client = TestClient(create_app(config=ServerConfig()))

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["speech_runtime"]["configured"] is True

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": os.getenv("IRODORI_MLX_MODEL_ID", "irodori-tts-mlx"),
            "input": os.getenv(
                "IRODORI_REAL_MLX_SMOKE_TEXT", "こんにちは。これは実行時スモークテストです。"
            ),
            "voice": "voicedesign",
            "response_format": "wav",
            "irodori": {
                "no_reference": True,
                "caption": os.getenv(
                    "IRODORI_REAL_MLX_SMOKE_CAPTION", "落ち着いた明瞭なナレーション"
                ),
                "preset": os.getenv("IRODORI_REAL_MLX_SMOKE_PRESET", "fast"),
                "chunking": False,
            },
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"

    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() >= 1
        assert wav_file.getsampwidth() > 0
        assert wav_file.getframerate() > 0
        assert wav_file.getnframes() > 0
        frames = wav_file.readframes(wav_file.getnframes())

    assert frames

from fastapi.testclient import TestClient

from irodori_tts_mlx_server import app


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["speech_runtime"] == {
        "runtime": "unconfigured",
        "configured": False,
        "loaded": False,
        "model_id": "irodori-tts-mlx",
    }
    assert response.json()["server"] == {
        "auth_enabled": False,
        "max_concurrent_synthesis": 1,
        "queue_timeout_seconds": 30.0,
    }

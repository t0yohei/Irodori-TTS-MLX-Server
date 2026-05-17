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
        "load_state": "unconfigured",
        "model_id": "irodori-tts-mlx",
    }
    server = response.json()["server"]
    assert server["auth_enabled"] is False
    assert server["max_concurrent_synthesis"] == 1
    assert server["queue_timeout_seconds"] == 30.0
    assert server["voices"]["dir"] == "voices"
    assert server["voices"]["files"] == 0
    assert server["voices"]["formats"] == [".wav"]

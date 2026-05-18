import importlib

import pytest
from fastapi.testclient import TestClient

from irodori_tts_mlx_server.runtime import IrodoriRuntimeConfig, runtime_config_from_env


class CapturedConfigRuntime:
    def __init__(self, config: IrodoriRuntimeConfig) -> None:
        self.config = config

    def list_models(self) -> list[str]:
        return [self.config.model_id]

    def status_metadata(self) -> dict[str, object]:
        return {
            "runtime": "captured",
            "configured": self.config.configured,
            "loaded": self.config.preload,
            "model_id": self.config.model_id,
            "weights_repo": self.config.weights_repo,
            "weights_revision": self.config.weights_revision,
        }

    def generate_speech(self, _request):
        raise AssertionError("test should not load model weights")


def test_cli_runtime_flags_configure_served_factory_app(monkeypatch) -> None:
    for name in (
        "IRODORI_MLX_PRELOAD",
        "IRODORI_MLX_WEIGHTS_REPO",
        "IRODORI_MLX_WEIGHTS_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)

    import irodori_tts_mlx_server

    package_app_response = TestClient(irodori_tts_mlx_server.app).get("/health")
    assert package_app_response.json()["speech_runtime"]["runtime"] == "unconfigured"

    main_module = importlib.import_module("irodori_tts_mlx_server.__main__")
    factory_module = importlib.import_module("irodori_tts_mlx_server.factory")
    served_health = {}

    def fake_create_default_runtime() -> CapturedConfigRuntime:
        return CapturedConfigRuntime(runtime_config_from_env())

    def fake_uvicorn_run(app_target: str, **kwargs) -> None:
        assert app_target == "irodori_tts_mlx_server.factory:create_app"
        assert kwargs["factory"] is True

        monkeypatch.setattr(factory_module, "create_default_runtime", fake_create_default_runtime)
        module_name, factory_name = app_target.split(":")
        factory = getattr(importlib.import_module(module_name), factory_name)
        response = TestClient(factory()).get("/health")
        served_health.update(response.json()["speech_runtime"])

    monkeypatch.setattr(main_module.uvicorn, "run", fake_uvicorn_run)

    main_module.main(["--weights-repo", "owner/repo", "--weights-revision", "main", "--preload"])

    assert served_health == {
        "runtime": "captured",
        "configured": True,
        "loaded": True,
        "model_id": "irodori-tts-mlx",
        "weights_repo": "owner/repo",
        "weights_revision": "main",
    }


def test_server_control_env_configures_served_factory_app(monkeypatch) -> None:
    monkeypatch.setenv("IRODORI_SERVER_BEARER_TOKEN", "local-secret")
    monkeypatch.setenv("IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS", "2")
    monkeypatch.setenv("IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS", "0.5")

    main_module = importlib.import_module("irodori_tts_mlx_server.__main__")
    factory_module = importlib.import_module("irodori_tts_mlx_server.factory")
    served_health = {}
    unauthorized_status = 0
    authorized_status = 0

    def fake_create_default_runtime() -> CapturedConfigRuntime:
        return CapturedConfigRuntime(runtime_config_from_env())

    def fake_uvicorn_run(app_target: str, **kwargs) -> None:
        nonlocal unauthorized_status, authorized_status
        assert app_target == "irodori_tts_mlx_server.factory:create_app"
        assert kwargs["factory"] is True

        monkeypatch.setattr(factory_module, "create_default_runtime", fake_create_default_runtime)
        module_name, factory_name = app_target.split(":")
        factory = getattr(importlib.import_module(module_name), factory_name)
        client = TestClient(factory())
        served_health.update(client.get("/health").json()["server"])
        unauthorized_status = client.get("/v1/models").status_code
        authorized_status = client.get(
            "/v1/models", headers={"Authorization": "Bearer local-secret"}
        ).status_code

    monkeypatch.setattr(main_module.uvicorn, "run", fake_uvicorn_run)

    main_module.main([])

    assert served_health["auth_enabled"] is True
    assert served_health["max_concurrent_synthesis"] == 2
    assert served_health["queue_timeout_seconds"] == 0.5
    assert served_health["voices"]["dir"] == "voices"
    assert served_health["voices"]["formats"] == [
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".ogg",
        ".opus",
        ".aac",
        ".webm",
    ]
    assert unauthorized_status == 401
    assert authorized_status == 200


def test_cli_rejects_conflicting_weight_sources() -> None:
    main_module = importlib.import_module("irodori_tts_mlx_server.__main__")

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(["--weights-dir", "/weights/layout", "--weights-repo", "owner/repo"])

    assert exc_info.value.code == 2

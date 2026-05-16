import importlib

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
    app_module = importlib.import_module("irodori_tts_mlx_server.app")
    served_health = {}

    def fake_create_default_runtime() -> CapturedConfigRuntime:
        return CapturedConfigRuntime(runtime_config_from_env())

    def fake_uvicorn_run(app_target: str, **kwargs) -> None:
        assert app_target == "irodori_tts_mlx_server.app:create_app"
        assert kwargs["factory"] is True

        monkeypatch.setattr(app_module, "create_default_runtime", fake_create_default_runtime)
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

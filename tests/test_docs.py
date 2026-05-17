from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_model_setup_doc_covers_required_weight_sources_and_smoke_commands() -> None:
    doc = (ROOT / "docs" / "real_model_setup.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert "docs/real_model_setup.md" in readme
    for required in (
        "IRODORI_MLX_WEIGHTS_REPO",
        "IRODORI_MLX_WEIGHTS_DIR",
        "IRODORI_MLX_WEIGHTS_PATH",
        "IRODORI_MLX_MODEL_CONFIG_JSON",
        "irodori_mlx_manifest.json",
        "model_config.json",
        "tokenizer_config.json",
        "conversion_metadata.json",
        "weights.npz",
        "curl http://127.0.0.1:8000/v1/audio/speech",
        "--output speech.wav",
        "runtime_unavailable",
        "server_configuration_error",
    ):
        assert required in doc


def test_deployment_doc_covers_operational_configuration() -> None:
    doc = (ROOT / "docs" / "deployment.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert "docs/deployment.md" in readme
    for required in (
        "Apple Silicon",
        "python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000",
        "IRODORI_SERVER_BEARER_TOKEN",
        "IRODORI_API_KEY",
        "Authorization: Bearer",
        "IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS",
        "IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS",
        "synthesis_queue_timeout",
        "curl --fail http://127.0.0.1:8000/health",
        "server.max_concurrent_synthesis",
        "runtime_unavailable",
        "server_configuration_error",
        "launchd",
        "StandardOutPath",
        "StandardErrorPath",
        "real_model_setup.md",
    ):
        assert required in doc

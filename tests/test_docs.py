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

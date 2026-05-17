from __future__ import annotations

import ast
import re
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


def test_openai_client_examples_cover_common_usage_paths() -> None:
    doc = (ROOT / "docs" / "openai_client_examples.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert "docs/openai_client_examples.md" in readme
    for required in (
        "curl http://127.0.0.1:8000/v1/models",
        "curl http://127.0.0.1:8000/v1/audio/speech",
        "Authorization: Bearer <token>",
        '"response_format": "wav"',
        '"response_format": "mp3"',
        "from openai import OpenAI",
        "base_url=",
        "IRODORI_API_KEY",
        "client.models.list()",
        "client.audio.speech.create(",
        "extra_body",
        'response_format="wav"',
        'response_format="flac"',
        "unsupported_streaming",
        "stream=true",
        "Accept: text/event-stream",
        "FFmpeg",
    ):
        assert required in doc


def test_openai_client_examples_python_snippets_parse() -> None:
    doc = (ROOT / "docs" / "openai_client_examples.md").read_text()

    snippets = re.findall(r"~~~python\n(.*?)\n~~~", doc, flags=re.DOTALL)
    assert snippets
    for snippet in snippets:
        ast.parse(snippet)


def test_upstream_compatibility_matrix_covers_required_gap_areas() -> None:
    doc = (ROOT / "docs" / "upstream_compatibility.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert "docs/upstream_compatibility.md" in readme
    for required in (
        "OpenAI-compatible model list",
        "Speech request base fields",
        "Response formats",
        "Reference voice support",
        "VoiceDesign caption / no-reference support",
        "Long-text chunking",
        "Queue / concurrency controls",
        "Bearer-token auth",
        "Docker / deployment support",
        "Voice management endpoints",
        "Runtime / model backend",
        "Implemented",
        "Partial",
        "Unsupported",
        "Intentionally out of scope",
        "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
    ):
        assert required in doc

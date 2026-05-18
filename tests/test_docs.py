from __future__ import annotations

import ast
import plistlib
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


def test_deployment_doc_covers_operational_configuration() -> None:
    doc = (ROOT / "docs" / "deployment.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert "docs/deployment.md" in readme
    for required in (
        "Apple Silicon",
        "python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000",
        "IRODORI_SERVER_BEARER_TOKEN",
        'test -n "$IRODORI_SERVER_BEARER_TOKEN"',
        "IRODORI_API_KEY",
        "Authorization: Bearer",
        "IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS",
        "IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS",
        "synthesis_queue_timeout",
        "curl --fail http://127.0.0.1:8000/health",
        "server.max_concurrent_synthesis",
        "server.status",
        "server.error.code",
        "runtime_unavailable",
        "server_configuration_error",
        "launchd",
        "deployment/dev.irodori.tts-mlx-server.plist.template",
        "deployment/local.env.example",
        "plutil -lint",
        "launchctl print",
        "kickstart",
        "Apple Silicon",
        "Docker and",
        "Compose are intentionally not the default path",
        "StandardOutPath",
        "StandardErrorPath",
        "real_model_setup.md",
    ):
        assert required in doc


def test_packaged_local_deployment_templates_stay_consistent() -> None:
    doc = (ROOT / "docs" / "deployment.md").read_text()
    plist_path = ROOT / "deployment" / "dev.irodori.tts-mlx-server.plist.template"
    env_path = ROOT / "deployment" / "local.env.example"

    assert plist_path.name in doc
    assert env_path.name in doc

    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["Label"] == "dev.irodori.tts-mlx-server"
    assert plist["ProgramArguments"] == [
        "/opt/irodori-tts-mlx-server/.venv/bin/irodori-tts-mlx-server",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["StandardOutPath"].endswith("/logs/server.log")
    assert plist["StandardErrorPath"].endswith("/logs/server.err.log")
    assert "/opt/irodori-tts-mlx-data/" in plist["StandardOutPath"]
    assert "/opt/irodori-tts-mlx-data/" in plist["StandardErrorPath"]

    environment = plist["EnvironmentVariables"]
    env_example = env_path.read_text()
    for required in (
        "IRODORI_MLX_WEIGHTS_DIR",
        "IRODORI_MLX_PRELOAD",
        "IRODORI_MLX_MODEL_ID",
        "IRODORI_MLX_CODEC_ARTIFACT_REPO",
        "IRODORI_MLX_CODEC_PATH",
        "IRODORI_MLX_CODEC_RUNTIME_MODE",
        "IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS",
        "IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS",
        "IRODORI_SERVER_VOICES_DIR",
        "IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES",
    ):
        assert required in environment
        assert required in env_example

    assert environment["IRODORI_SERVER_VOICES_DIR"] == "/opt/irodori-tts-mlx-data/voices"
    assert "/opt/irodori-tts-mlx-server/voices" not in env_example
    assert "/opt/irodori-tts-mlx-server/local.env" not in doc

    for required in (
        "IRODORI_SERVER_BEARER_TOKEN",
        "IRODORI_API_KEY",
        "IRODORI_MLX_CODEC_ARTIFACT_REPO",
        "IRODORI_MLX_CODEC_PATH",
        "IRODORI_MLX_CODEC_DEVICE",
        "IRODORI_MLX_CODEC_RUNTIME_MODE=mlx",
    ):
        assert required in env_example


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
        "Managed reference voice scope",
        "Runtime / model backend",
        "Implemented",
        "Partial",
        "Unsupported",
        "Intentionally out of scope",
        "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
    ):
        assert required in doc


def test_readme_covers_managed_reference_voice_scope() -> None:
    readme = (ROOT / "README.md").read_text()

    for required in (
        "IRODORI_SERVER_VOICES_DIR",
        "GET /v1/audio/voices",
        "POST /v1/audio/voices",
        "PUT /v1/audio/voices/{voice_id}",
        "DELETE /v1/audio/voices/{voice_id}",
        "reference voices may use",
        "arbitrary local paths are rejected",
    ):
        assert required in readme

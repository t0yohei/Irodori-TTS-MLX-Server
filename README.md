# Irodori-TTS-MLX-Server

OpenAI-compatible local TTS server for
[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX).

The initial implementation is intentionally scoped to a small MVP: expose
`POST /v1/audio/speech` for VoiceDesign v2 no-reference/caption generation
through an MLX-backed runtime. See [docs/mvp_scope.md](docs/mvp_scope.md) for
the current API target, non-goals, and follow-up implementation boundaries.

This scaffold currently includes the package bootstrap, local development
workflow, and `GET /health`. Model listing, speech generation, runtime adapters,
authentication, and audio conversion are tracked in later MVP issues.

## Local Development

Create a virtual environment and install the package with development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Run the local server:

```bash
python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000
```

The installed console script is equivalent:

```bash
irodori-tts-mlx-server --host 127.0.0.1 --port 8000
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Validation

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

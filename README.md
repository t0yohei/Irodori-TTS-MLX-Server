# Irodori-TTS-MLX-Server

FastAPI server package for exposing Irodori-TTS-MLX through an HTTP API.

This initial scaffold includes the package bootstrap, local development
workflow, `GET /health`, and the OpenAI-compatible MVP routes:

- `GET /v1/models`
- `POST /v1/audio/speech`

The default runtime is import-safe without model weights. It lists the MVP model
id but returns a clear `runtime_unavailable` error for speech generation until a
real Irodori-TTS-MLX runtime adapter is configured.

## Local development

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

List OpenAI-compatible models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Generate WAV speech once a runtime adapter is configured:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts-mlx","input":"hello","voice":"alloy","response_format":"wav"}' \
  --output speech.wav
```

Requests accept `model`, `input`, `voice`, `response_format`, `speed`, and
an `irodori` options object. `response_format=wav` is supported for the MVP.
Streaming responses are not supported and return an OpenAI-style error object.

## Validation

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

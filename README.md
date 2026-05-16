# Irodori-TTS-MLX-Server

OpenAI-compatible local TTS server for
[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX).

The initial implementation is intentionally scoped to a small MVP: expose
`POST /v1/audio/speech` for VoiceDesign v2 no-reference/caption generation
through an MLX-backed runtime. See [docs/mvp_scope.md](docs/mvp_scope.md) for
the current API target, non-goals, and follow-up implementation boundaries.

This scaffold includes the package bootstrap, local development workflow,
`GET /health`, and the OpenAI-compatible MVP routes:

- `GET /v1/models`
- `POST /v1/audio/speech`

The default runtime is import-safe without model weights. It lists the MVP model
id but returns a clear `runtime_unavailable` error for speech generation until a
converted Irodori-TTS-MLX weights are configured.

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
{"status":"ok","speech_runtime":{"runtime":"unconfigured","configured":false,"loaded":false,"model_id":"irodori-tts-mlx"}}
```

Configure the real MLX runtime with either a hosted converted-weights layout:

```bash
python -m irodori_tts_mlx_server \
  --weights-repo owner/irodori-tts-mlx-converted-weights \
  --host 127.0.0.1 \
  --port 8000
```

or local converted weights plus the matching model config:

```bash
python -m irodori_tts_mlx_server \
  --weights /path/to/irodori-tts-mlx.npz \
  --model-config-json /path/to/model_config.json
```

The same settings can be supplied through environment variables:
`IRODORI_MLX_WEIGHTS_REPO`, `IRODORI_MLX_WEIGHTS_DIR`,
`IRODORI_MLX_WEIGHTS_PATH`, and `IRODORI_MLX_MODEL_CONFIG_JSON`.
`IRODORI_MLX_PRELOAD=1` loads the model during startup; otherwise loading is
lazy on the first speech request.

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
an `irodori` options object. `response_format=wav` and `response_format=pcm`
work without optional encoders. `mp3`, `flac`, `opus`, and `aac` use FFmpeg
when it is installed; otherwise the server returns an OpenAI-style error with
setup guidance. Streaming responses are not supported and return an
OpenAI-style error object.

Supported `irodori` runtime options include `reference_wav`, `no_reference`,
`caption`, `seconds`, `duration_scale`, `num_steps`, `seed`, `cfg_scale_text`,
`cfg_scale_caption`, `cfg_scale_speaker`, `cfg_guidance_mode`, `cfg_min_t`,
`cfg_max_t`, `max_reference_seconds`, and `no_context_kv_cache`. When
`duration_scale` is omitted, OpenAI `speed` maps to `duration_scale=1/speed`.

## Validation

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

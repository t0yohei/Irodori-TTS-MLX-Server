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

## Relationship to Aratako/Irodori-TTS-Server

[Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) is
the upstream OpenAI-compatible server for
[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS). It targets the
PyTorch Irodori-TTS 500M v3 base model and includes a broader production server
surface: reference voice management, long-text chunking, request queueing,
bearer-token auth, dynamic LoRA options, and response formats such as `wav`,
`mp3`, `flac`, `opus`, `aac`, and `pcm`.

This repository is a separate MLX-focused server for
[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX). The compatibility
goal is to preserve the OpenAI-style client shape where it maps cleanly to the
MLX runtime, starting with `GET /v1/models` and `POST /v1/audio/speech`, while
keeping the backend boundary narrow enough to run on Apple Silicon with
converted MLX weights.

Choose the upstream server when you want the PyTorch v3 base model, CUDA-oriented
deployment, Docker Compose setup, built-in voice-file APIs, automatic long-text
chunking, queue controls, optional auth, or multiple encoded response formats.
Choose this server when you specifically want to exercise Irodori-TTS-MLX on
Apple Silicon and can provide converted MLX weights plus the matching model
configuration.

The current MLX server MVP does not claim parity with the upstream server. The
initial surface supports the OpenAI-compatible model list and speech endpoint,
VoiceDesign v2 no-reference/caption options, `wav` output, and clear
configuration errors before weights are available. Runtime smoke evidence,
compressed audio formats, voice management endpoints, long-text chunking,
queueing, bearer-token auth, and full upstream option coverage are tracked as
follow-up work unless explicitly documented as implemented here.

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
{"status":"ok","speech_runtime":{"runtime":"unconfigured","configured":false,"loaded":false,"model_id":"irodori-tts-mlx"},"server":{"auth_enabled":false,"max_concurrent_synthesis":1,"queue_timeout_seconds":30.0}}
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

Local development does not require authentication by default. Set
`IRODORI_SERVER_BEARER_TOKEN` to require `Authorization: Bearer <token>` on the
OpenAI-compatible `/v1/*` routes while leaving `/health` unauthenticated for
local probes. `IRODORI_API_KEY` is accepted as a compatibility alias when
`IRODORI_SERVER_BEARER_TOKEN` is unset.

Speech generation is bounded by a synthesis semaphore so expensive MLX work does
not run concurrently unless configured. Defaults are conservative:
`IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS=1` and
`IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS=30`. Requests that cannot acquire a slot
before the timeout return a 503 OpenAI-style `synthesis_queue_timeout` error.

List OpenAI-compatible models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Generate WAV speech once a runtime adapter is configured:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"model":"irodori-tts-mlx","input":"hello","voice":"voicedesign","response_format":"wav","irodori":{"no_reference":true,"caption":"calm narration, clear diction","preset":"balanced"}}' \
  --output speech.wav
```

Requests accept `model`, `input`, `voice`, `response_format`, `speed`, and
an `irodori` options object. `response_format=wav` and `response_format=pcm`
work without optional encoders. `mp3`, `flac`, `opus`, and `aac` use FFmpeg
when it is installed; otherwise the server returns an OpenAI-style error with
setup guidance. Streaming responses are not supported and return an
OpenAI-style error object.

Supported `irodori` runtime options include `reference_wav`, `no_reference`,
`caption`, `preset`, `seconds`, `duration_scale`, `num_steps`, `seed`,
`cfg_scale_text`, `cfg_scale_caption`, `cfg_scale_speaker`, `cfg_guidance_mode`,
`cfg_min_t`, `cfg_max_t`, `max_reference_seconds`, `no_context_kv_cache`,
`chunking`, `chunk_max_chars`, `tail_trim_ms`, `tail_silence_trim_ms`,
`tail_silence_keep_ms`, and `tail_silence_threshold`.
When `duration_scale` is omitted, OpenAI `speed` maps to
`duration_scale=1/speed`.

Long text chunking is enabled by default. The server splits text on Japanese and
English punctuation before falling back to hard character slices, using
`IRODORI_MLX_TEXT_MAX_LENGTH` as the default `chunk_max_chars`. Set
`irodori.chunking=false` to send the full text to the runtime unchanged, or set
`irodori.chunk_max_chars` to tune the split point. When `irodori.seconds` is
omitted, each chunk also omits `seconds` so Irodori-TTS-MLX can use its duration
fallback or predicted-duration behavior. When `irodori.seconds` is explicit, the
server distributes that total duration across chunks by character count.

Tail artifact controls are optional and run before chunk concatenation.
`tail_trim_ms` removes a fixed amount from the end of each generated chunk.
`tail_silence_trim_ms` removes trailing 16-bit PCM silence only when at least
that much silence is present, preserving `tail_silence_keep_ms`; the silence
threshold defaults to `256` and can be tuned with `tail_silence_threshold`.

For VoiceDesign v2 caption-conditioned hosted weights, set
`irodori.no_reference=true` and provide a concise style caption such as
`"calm narration, clear diction"` or `"bright young voice, energetic delivery"`.
`irodori.preset` accepts `fast`, `balanced`, or `quality`, mapping to 12, 24, or
40 sampling steps. An explicit `irodori.num_steps` overrides the preset. Do not
set `irodori.reference_wav` together with `irodori.no_reference=true`; if
`irodori.no_reference=false`, a `reference_wav` path is required.

## Validation

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

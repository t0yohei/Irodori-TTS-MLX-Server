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
- `GET`, `POST`, `GET by id`, `PUT`, and `DELETE` for
  `/v1/audio/voices`

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
deployment, Docker Compose setup, broad voice-file and alias APIs, automatic
long-text chunking, queue controls, optional auth, or multiple encoded response formats.
Choose this server when you specifically want to exercise Irodori-TTS-MLX on
Apple Silicon and can provide converted MLX weights plus the matching model
configuration.

The current MLX server MVP does not claim parity with the upstream server. The
initial surface supports the OpenAI-compatible model list and speech endpoint,
VoiceDesign v2 no-reference/caption options, managed WAV reference uploads,
`wav` output, and clear configuration errors before weights are available.
Runtime smoke evidence, compressed audio formats, long-text chunking, queueing,
bearer-token auth, and full upstream option coverage are tracked as follow-up
work unless explicitly documented as implemented here.

See [docs/upstream_compatibility.md](docs/upstream_compatibility.md) for the
concrete upstream compatibility gap matrix.

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
{"status":"ok","speech_runtime":{"runtime":"unconfigured","configured":false,"loaded":false,"load_state":"unconfigured","model_id":"irodori-tts-mlx"},"server":{"auth_enabled":false,"max_concurrent_synthesis":1,"queue_timeout_seconds":30.0}}
```

See [docs/real_model_setup.md](docs/real_model_setup.md) for the full fresh
checkout setup walkthrough, converted-weights layout contract, local run
commands, speech `curl` example, and common error codes.
See [docs/openai_client_examples.md](docs/openai_client_examples.md) for
OpenAI-compatible `curl` and Python client examples, including bearer auth,
FFmpeg-backed response formats, and unsupported streaming behavior.
See [docs/deployment.md](docs/deployment.md) for production-ish local deployment
guidance covering Apple Silicon host assumptions, bearer auth, queue controls,
health checks, logs, and the packaged launchd templates under
[deployment/](deployment/).

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
lazy on the first speech request. `/health` reports `speech_runtime.load_state`
as `unconfigured`, `not_loaded`, `loading`, `loaded`, or `failed` without
forcing model load.

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

Reference voice files can be managed through a small upstream-style subset under
`/v1/audio/voices`. Set `IRODORI_SERVER_VOICES_DIR` to choose the storage
directory; it defaults to `voices` relative to the server process. Managed
reference voices are WAV-only. Voice IDs may contain ASCII letters, numbers,
underscores, or hyphens. Uploaded files are always stored as
`<voice_id>.wav` inside that directory, and speech requests that set
`voice` to a managed ID automatically receive `irodori.reference_wav` and
`irodori.no_reference=false` unless the request already supplies explicit
reference options or `irodori.no_reference=true`. Upstream-style `no_ref=false`
still allows managed voice resolution. This route does not resolve arbitrary
client-supplied file paths, alias files, latent references, or remote URLs.
Uploads are capped by `IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES`, which defaults
to 50 MiB. New managed voice files are committed atomically so interrupted
uploads do not leave partial `<voice_id>.wav` files behind.

Security note: the management API does not resolve arbitrary client-supplied file paths.

List and upload managed voices:

```bash
curl http://127.0.0.1:8000/v1/audio/voices \
  -H 'Authorization: Bearer <token>'

curl http://127.0.0.1:8000/v1/audio/voices \
  -H 'Authorization: Bearer <token>' \
  -F voice_id=sample \
  -F file=@sample.wav
```

Additional managed voice routes are `GET /v1/audio/voices/{voice_id}`,
`PUT /v1/audio/voices/{voice_id}`, and
`DELETE /v1/audio/voices/{voice_id}`. Use the upstream PyTorch server when you
need alias-file resolution, latent reference files, or non-WAV voice upload
formats.

Managed voice endpoint summary:

- `GET /v1/audio/voices`
- `POST /v1/audio/voices`
- `GET /v1/audio/voices/{voice_id}`
- `PUT /v1/audio/voices/{voice_id}`
- `DELETE /v1/audio/voices/{voice_id}`

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
an `irodori` options object. For upstream client compatibility, unambiguous
top-level runtime option aliases such as `no_ref`, `seconds`, `duration_scale`,
`num_steps`, `seed`, `cfg_scale_*`, `max_ref_seconds`, `context_kv_cache`, and
`chunking_enabled` are normalized into `irodori`. `response_format=wav` and
`response_format=pcm`
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
The upstream `irodori` aliases `ref_wav`, `no_ref`, `max_ref_seconds`,
`context_kv_cache`, and `chunking_enabled` are also accepted when they map
cleanly to the MLX runtime. Unsupported or ambiguous upstream options such as
`lora_adapter`, `ref_latent`, `chunk_min_chars`, and PyTorch schedule-specific
controls are rejected instead of being silently ignored.
When `duration_scale` is omitted, OpenAI `speed` maps to
`duration_scale=1/speed`.

`irodori.lora_adapter` is intentionally unsupported with the current
Irodori-TTS-MLX runtime boundary. Blank values are ignored, but non-empty values
return `invalid_irodori_options`; path-like values are rejected explicitly so
requests cannot load arbitrary local adapter files. LoRA support should be added
only after the MLX runtime exposes an adapter field plus clear alias/allowlist
and cache/reload semantics.

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

### Opt-in Real MLX Runtime Smoke

Normal `pytest` runs do not require model weights, the Irodori-TTS-MLX runtime
package, upstream Irodori-TTS, tokenizer assets, or codec artifacts. The real
runtime smoke test is skipped unless explicitly enabled and configured.

The shortest currently documented converted-weights source for VoiceDesign
no-reference/caption smoke testing is the approved hosted layout
`t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign`, as documented by
Irodori-TTS-MLX's hosted weights usage guide. Equivalent local hosted layouts can
be supplied with `IRODORI_MLX_WEIGHTS_DIR`, or direct converted `.npz` weights
with `IRODORI_MLX_WEIGHTS_PATH` plus `IRODORI_MLX_MODEL_CONFIG_JSON`.

Install the server, development tools, Irodori-TTS-MLX runtime extras, and the
upstream Irodori-TTS package in the same environment before running the smoke:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e /path/to/Irodori-TTS-MLX"[runtime]"
python -m pip install -e /path/to/Irodori-TTS
```

Run the opt-in pytest smoke with hosted converted VoiceDesign weights:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
IRODORI_REAL_MLX_SMOKE=1 \
IRODORI_MLX_WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
IRODORI_MLX_CODEC_RUNTIME_MODE=persistent \
pytest -m real_mlx tests/test_real_mlx_smoke.py
```

The test constructs the FastAPI app with the real runtime configuration, calls
`POST /v1/audio/speech` with `voice=voicedesign`,
`irodori.no_reference=true`, and a caption, then verifies the response is valid
WAV audio with non-empty PCM frames.

To validate through a running local server instead of pytest:

```bash
PYTHONPATH=/path/to/Irodori-TTS:${PYTHONPATH:-} \
IRODORI_MLX_WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
IRODORI_MLX_CODEC_RUNTIME_MODE=persistent \
python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts-mlx","input":"こんにちは。これは実行時スモークテストです。","voice":"voicedesign","response_format":"wav","irodori":{"no_reference":true,"caption":"落ち着いた明瞭なナレーション","preset":"fast","chunking":false}}' \
  --output /tmp/irodori-real-mlx-smoke.wav

python - <<'PY'
from pathlib import Path
from wave import open as open_wav

with open_wav(str(Path("/tmp/irodori-real-mlx-smoke.wav")), "rb") as wav_file:
    assert wav_file.getnframes() > 0
    assert wav_file.readframes(wav_file.getnframes())
PY
```

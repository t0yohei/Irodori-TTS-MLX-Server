# Production Deployment Guidance

This guide covers a production-ish local deployment for an Apple Silicon host:
a long-running process bound to localhost or a trusted private interface,
optional bearer-token auth for OpenAI-compatible routes, conservative local
concurrency, and explicit health checks. It assumes the real runtime setup from
[real_model_setup.md](real_model_setup.md) is already complete; keep converted
weights and runtime dependency details there instead of duplicating them here.

The packaged deployment target for this repository is a macOS launchd service
template plus an environment example for local Apple Silicon hosts. Docker and
Compose are intentionally not the default path: MLX expects the macOS/Apple
Silicon runtime and this project should not imply CUDA or PyTorch container
support.

## Recommended Launch Command

Run the server from a dedicated virtual environment and pin the model artifact
source. For a private single-host setup, bind to localhost and put any remote
access behind SSH forwarding or a reverse proxy that you control:

```bash
cd /opt/irodori-tts-mlx-server
source .venv/bin/activate

export IRODORI_MLX_WEIGHTS_DIR=/opt/irodori-models/voicedesign-v2
export IRODORI_MLX_PRELOAD=1
IRODORI_SERVER_BEARER_TOKEN="$(security find-generic-password -a irodori-tts-mlx-server -s irodori-server-token -w)"
test -n "$IRODORI_SERVER_BEARER_TOKEN"
export IRODORI_SERVER_BEARER_TOKEN
export IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS=1
export IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS=60

exec python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000
```

Use `IRODORI_MLX_WEIGHTS_REPO` with `IRODORI_MLX_WEIGHTS_REVISION` when the
converted weights live in an approved hosted repository, or
`IRODORI_MLX_WEIGHTS_DIR` for a local copy of the same hosted layout. See
[real_model_setup.md](real_model_setup.md) for the full
artifact layout contract and speech smoke command.

## Environment Variables

Required for real synthesis, choose exactly one weight source:

| Variable | Required when | Notes |
| --- | --- | --- |
| `IRODORI_MLX_WEIGHTS_REPO` | Hosted converted weights | Repository id such as `owner/repo`. Pin `IRODORI_MLX_WEIGHTS_REVISION` for repeatable deployments. |
| `IRODORI_MLX_WEIGHTS_DIR` | Local hosted-layout directory | Directory containing `irodori_mlx_manifest.json`, `model_config.json`, tokenizer metadata, converted weights, and checksums. |

Optional runtime and server controls:

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_MLX_WEIGHTS_REVISION` | provider default | Branch, tag, or commit for hosted weights. |
| `IRODORI_MLX_MODEL_ID` | `irodori-tts-mlx` | Model id returned by `/v1/models` and accepted by `/v1/audio/speech`. |
| `IRODORI_MLX_PRELOAD` | unset / false | Set to `1` to load the model during startup so failures are visible before serving traffic. |
| `IRODORI_MLX_TEXT_MAX_LENGTH` | `256` | Default long-text chunk size used by runtime request mapping. |
| `IRODORI_MLX_CAPTION_MAX_LENGTH` | runtime default | Optional caption-token limit. |
| `IRODORI_MLX_CODEC_REPO` | `Aratako/Semantic-DACVAE-Japanese-32dim` | Codec repository used by the Irodori-TTS-MLX runtime. |
| `IRODORI_MLX_CODEC_ARTIFACT_REPO` | `t0yohei/Irodori-TTS-MLX-DACVAE-Codec` | Hosted MLX DACVAE codec artifact repo used by `mlx` and `mlx-decode` when `IRODORI_MLX_CODEC_PATH` is unset. |
| `IRODORI_MLX_CODEC_ARTIFACT_REVISION` | provider default | Optional revision for the hosted codec artifact repo. |
| `IRODORI_MLX_CODEC_PATH` | unset | Optional local MLX DACVAE codec artifact path. Overrides the hosted codec artifact repo. |
| `IRODORI_MLX_CODEC_DEVICE` | `cpu` | Keep on `cpu` unless the runtime has validated another device for this host. |
| `IRODORI_MLX_CODEC_RUNTIME_MODE` | `persistent` | `persistent` uses the upstream PyTorch codec bridge. `mlx-decode` uses MLX DACVAE decode with the reference encoder fallback. `mlx` uses the MLX DACVAE artifact for both sides when the artifact supports it. |
| `IRODORI_MLX_ENABLE_WATERMARK` | unset / false | Enables runtime watermarking when supported by the runtime. |
| `IRODORI_MLX_DISABLE_CODEC_NORMALIZE` | unset / false | Disables codec audio normalization. |
| `IRODORI_SERVER_BEARER_TOKEN` | unset | Enables bearer-token auth for `/v1/*` routes. Prefer a secret manager or macOS Keychain lookup over a checked-in `.env`. |
| `IRODORI_API_KEY` | unset | Compatibility alias used only when `IRODORI_SERVER_BEARER_TOKEN` is unset. |
| `IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS` | `1` | Number of synthesis requests allowed to run at once. Keep `1` for most Apple Silicon hosts. |
| `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS` | `30` | Seconds a request waits for a synthesis slot before returning `synthesis_queue_timeout`; `0` disables waiting. |
| `IRODORI_SERVER_VOICES_DIR` | `voices` | Directory used by `/v1/audio/voices` for managed WAV reference uploads. Keep this outside the repository for long-running deployments. |
| `IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES` | `52428800` | Maximum accepted managed voice upload size before returning `voice_file_too_large`. |

The checked-in [deployment/local.env.example](../deployment/local.env.example)
contains the same operational knobs in copyable form. Keep the edited copy
outside the repository, especially when adding bearer tokens or host-specific
paths.

## Auth and Network Binding

`/health` is intentionally unauthenticated so launchd, reverse proxies, and
local probes can verify process health. `IRODORI_SERVER_BEARER_TOKEN` protects
`/v1/models` and `/v1/audio/speech`:

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $IRODORI_SERVER_BEARER_TOKEN"
```

Bind to `127.0.0.1` for local OpenAI-compatible clients on the same Mac. If a
LAN or reverse-proxy deployment needs `0.0.0.0`, configure bearer auth and keep
TLS, firewalling, request limits, and access logs at the proxy layer.

## Concurrency and Queueing

MLX synthesis is memory- and compute-heavy. Start with:

```bash
IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS=1
IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS=60
```

Raise concurrency only after observing memory pressure, swap, and request
latency on the target Mac. When every slot is busy past the queue timeout, the
server returns HTTP 503 with OpenAI-style code `synthesis_queue_timeout`.
Clients should retry with backoff rather than opening parallel replacement
requests.

## Health Checks

Use `/health` for process and runtime visibility:

```bash
curl --fail http://127.0.0.1:8000/health
```

The response includes:

- `status`: process-level health, normally `ok`.
- `speech_runtime.configured`: whether a weights source is configured.
- `speech_runtime.loaded`: whether the runtime has loaded successfully.
- `speech_runtime.load_state`: one of `unconfigured`, `not_loaded`, `loading`,
  `loaded`, or `failed`; this does not force a lazy model load.
- `speech_runtime.last_load_error`: present when configuration or runtime load
  failed.
- `server.auth_enabled`: whether `/v1/*` bearer auth is enabled.
- `server.max_concurrent_synthesis` and `server.queue_timeout_seconds`: active
  queue controls.
- `server.status: "configuration_error"` with `server.error.code:
  "server_configuration_error"`: present when server env values are invalid.

For deployment checks, combine `/health` with a real WAV smoke request from
[real_model_setup.md](real_model_setup.md). `/health` does not synthesize audio
and does not prove the model can generate speech.

## Apple Silicon and MLX Runtime Assumptions

This server targets local Apple Silicon MLX usage. Run it on macOS with a Python
version supported by the package and install Irodori-TTS-MLX into the same
virtual environment as this server. Keep the process on a machine with enough
unified memory for the converted weights, tokenizer assets, codec runtime, and
peak request audio buffers.

Preloading is recommended for unattended processes because it exercises missing
dependency, missing artifact, and invalid config checks before the first client
request. If preload fails, the process still serves `/health` and returns
`runtime_unavailable` for synthesis so supervisors and operators can inspect the
failure. Lazy loading is useful during development, but the first speech request
will pay the load cost and may surface setup errors to the client.

## Logs and Common Failures

Uvicorn writes access logs and application errors to stdout/stderr. The server
also emits consistent application log events for `request_start`, `request_end`,
`runtime_load_start`, `runtime_load_complete`, `runtime_load_failed`,
`synthesis_queue_timeout`, and `generation_failed`. These logs include method,
path, status, timing, queue/runtime state, and failure class, but not request
text or bearer tokens. Under launchd, redirect them to files under a writable log
directory and rotate them with your normal host policy. Avoid sharing logs that
contain private host paths or operational secrets.

Common startup and runtime failures:

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `/health` shows `runtime: unconfigured` | No weights source was set. | Configure one supported weight source from [real_model_setup.md](real_model_setup.md). |
| `/health` shows `speech_runtime.load_state: "loading"` | The first request or preload is currently building the MLX runtime. | Wait for `loaded` or inspect logs if it changes to `failed`. |
| `/health` shows `speech_runtime.load_state: "failed"` | Runtime dependency, artifact, or model config loading failed. | Inspect `speech_runtime.last_load_error`, check `runtime_load_failed` logs, and rerun the real setup smoke. |
| `/health` shows `server.status: "configuration_error"` | Server env parsing failed, commonly an invalid concurrency or queue timeout value. | Check `server.error.message`; fix integer, boolean, or missing server config values. |
| `/v1/*` returns `server_configuration_error` | Server env parsing failed, commonly an invalid concurrency or queue timeout value. | Fix `IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS` or `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS`, then restart. |
| `/v1/*` returns `invalid_api_key` | Bearer token is enabled but missing or wrong. | Send `Authorization: Bearer <token>` and verify the active token source. |
| `/v1/audio/speech` returns `runtime_unavailable` | Irodori-TTS-MLX deps, weights, or model config failed at load time. | Run with `IRODORI_MLX_PRELOAD=1`, inspect logs, and rerun the real setup smoke. |
| `/v1/audio/speech` returns `synthesis_queue_timeout` | All synthesis slots are busy. | Increase client backoff or queue timeout before raising concurrency. |
| Process exits immediately under launchd | Bad working directory, venv path, or env file. | Run the launch command manually as the same user, then check launchd stderr. |

## Packaged Local launchd Deployment

Use launchd when the server should start at login for one macOS user. The
checked-in template is
[deployment/dev.irodori.tts-mlx-server.plist.template](../deployment/dev.irodori.tts-mlx-server.plist.template).
It uses the installed `irodori-tts-mlx-server` console script, binds to
`127.0.0.1:8000`, keeps MLX concurrency at `1`, preloads weights, and stores
managed WAV voices and logs outside the repository.

First run on a fresh host:

```bash
sudo mkdir -p /opt/irodori-tts-mlx-server /opt/irodori-tts-mlx-data /opt/irodori-models
sudo chown -R "$USER":staff /opt/irodori-tts-mlx-server /opt/irodori-tts-mlx-data /opt/irodori-models
git clone https://github.com/t0yohei/Irodori-TTS-MLX-Server.git /opt/irodori-tts-mlx-server
cd /opt/irodori-tts-mlx-server
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Install the external Irodori-TTS-MLX runtime package and place converted weights
using [real_model_setup.md](real_model_setup.md). Then copy and edit the local
templates:

```bash
mkdir -p /opt/irodori-tts-mlx-data/voices /opt/irodori-tts-mlx-data/logs
mkdir -p ~/Library/LaunchAgents
cp deployment/local.env.example /opt/irodori-tts-mlx-data/local.env
cp deployment/dev.irodori.tts-mlx-server.plist.template \
  ~/Library/LaunchAgents/dev.irodori.tts-mlx-server.plist
plutil -lint ~/Library/LaunchAgents/dev.irodori.tts-mlx-server.plist
```

launchd does not read shell-style `.env` files directly. Keep
`local.env.example` as the source of truth for host values, then copy those
values into the plist `EnvironmentVariables` dictionary or use a small private
wrapper script that exports the file before `exec`-ing the console script. Do
not store bearer tokens in the checked-in template; load
`IRODORI_SERVER_BEARER_TOKEN` from macOS Keychain in the wrapper when auth is
enabled.

Before loading launchd, validate the same command manually:

```bash
IRODORI_MLX_WEIGHTS_DIR=/opt/irodori-models/voicedesign-v2 \
IRODORI_MLX_PRELOAD=1 \
IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS=1 \
IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS=60 \
IRODORI_SERVER_VOICES_DIR=/opt/irodori-tts-mlx-data/voices \
/opt/irodori-tts-mlx-server/.venv/bin/irodori-tts-mlx-server --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
```

Load and restart the service:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.irodori.tts-mlx-server.plist
launchctl kickstart -k gui/$(id -u)/dev.irodori.tts-mlx-server
curl --fail http://127.0.0.1:8000/health
```

Inspect logs and service state:

```bash
launchctl print gui/$(id -u)/dev.irodori.tts-mlx-server
tail -n 100 /opt/irodori-tts-mlx-data/logs/server.err.log
tail -n 100 /opt/irodori-tts-mlx-data/logs/server.log
```

The plist template writes stdout and stderr through `StandardOutPath` and
`StandardErrorPath`; keep those paths writable by the launchd user.

Upgrade or restart after changing package versions, weights, or environment:

```bash
cd /opt/irodori-tts-mlx-server
git pull --ff-only
source .venv/bin/activate
python -m pip install -U -e .
launchctl kickstart -k gui/$(id -u)/dev.irodori.tts-mlx-server
```

Failure recovery:

- If `plutil -lint` fails, fix the plist before loading it.
- If launchd exits immediately, run the manual command as the same user and
  check `server.err.log`.
- If `/health` reports `configuration_error`, fix the invalid env value and
  kickstart the service.
- If `/health` reports a runtime load error, rerun the real model setup smoke
  before changing service settings.

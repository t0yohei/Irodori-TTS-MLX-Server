# Production Deployment Guidance

This guide covers a production-ish local deployment for an Apple Silicon host:
a long-running process bound to localhost or a trusted private interface,
optional bearer-token auth for OpenAI-compatible routes, conservative local
concurrency, and explicit health checks. It assumes the real runtime setup from
[real_model_setup.md](real_model_setup.md) is already complete; keep converted
weights and runtime dependency details there instead of duplicating them here.

## Recommended Launch Command

Run the server from a dedicated virtual environment and pin the model artifact
source. For a private single-host setup, bind to localhost and put any remote
access behind SSH forwarding or a reverse proxy that you control:

```bash
cd /opt/irodori-tts-mlx-server
source .venv/bin/activate

export IRODORI_MLX_WEIGHTS_DIR=/opt/irodori-models/voicedesign-v2
export IRODORI_MLX_PRELOAD=1
export IRODORI_SERVER_BEARER_TOKEN="$(security find-generic-password -a irodori-tts-mlx-server -s irodori-server-token -w)"
export IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS=1
export IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS=60

exec python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000
```

Use `IRODORI_MLX_WEIGHTS_REPO` with `IRODORI_MLX_WEIGHTS_REVISION` when the
converted weights live in an approved hosted repository, or
`IRODORI_MLX_WEIGHTS_PATH` with `IRODORI_MLX_MODEL_CONFIG_JSON` for a direct
local `.npz` file. See [real_model_setup.md](real_model_setup.md) for the full
artifact layout contract and speech smoke command.

## Environment Variables

Required for real synthesis, choose exactly one weight source:

| Variable | Required when | Notes |
| --- | --- | --- |
| `IRODORI_MLX_WEIGHTS_REPO` | Hosted converted weights | Repository id such as `owner/repo`. Pin `IRODORI_MLX_WEIGHTS_REVISION` for repeatable deployments. |
| `IRODORI_MLX_WEIGHTS_DIR` | Local hosted-layout directory | Directory containing `irodori_mlx_manifest.json`, `model_config.json`, tokenizer metadata, converted weights, and checksums. |
| `IRODORI_MLX_WEIGHTS_PATH` | Direct file source | Path to converted `weights.npz`; requires `IRODORI_MLX_MODEL_CONFIG_JSON`. |
| `IRODORI_MLX_MODEL_CONFIG_JSON` | Direct file source | Matching config for `IRODORI_MLX_WEIGHTS_PATH`. |

Optional runtime and server controls:

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_MLX_WEIGHTS_REVISION` | provider default | Branch, tag, or commit for hosted weights. |
| `IRODORI_MLX_MODEL_ID` | `irodori-tts-mlx` | Model id returned by `/v1/models` and accepted by `/v1/audio/speech`. |
| `IRODORI_MLX_PRELOAD` | unset / false | Set to `1` to load the model during startup so failures are visible before serving traffic. |
| `IRODORI_MLX_TEXT_MAX_LENGTH` | `256` | Default long-text chunk size used by runtime request mapping. |
| `IRODORI_MLX_CAPTION_MAX_LENGTH` | runtime default | Optional caption-token limit. |
| `IRODORI_MLX_CODEC_REPO` | `Aratako/Semantic-DACVAE-Japanese-32dim` | Codec repository used by the Irodori-TTS-MLX runtime. |
| `IRODORI_MLX_CODEC_DEVICE` | `cpu` | Keep on `cpu` unless the runtime has validated another device for this host. |
| `IRODORI_MLX_CODEC_RUNTIME_MODE` | `persistent` | Keeps codec startup cost out of repeated requests. |
| `IRODORI_MLX_ENABLE_WATERMARK` | unset / false | Enables runtime watermarking when supported by the runtime. |
| `IRODORI_MLX_DISABLE_CODEC_NORMALIZE` | unset / false | Disables codec audio normalization. |
| `IRODORI_SERVER_BEARER_TOKEN` | unset | Enables bearer-token auth for `/v1/*` routes. Prefer a secret manager or macOS Keychain lookup over a checked-in `.env`. |
| `IRODORI_API_KEY` | unset | Compatibility alias used only when `IRODORI_SERVER_BEARER_TOKEN` is unset. |
| `IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS` | `1` | Number of synthesis requests allowed to run at once. Keep `1` for most Apple Silicon hosts. |
| `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS` | `30` | Seconds a request waits for a synthesis slot before returning `synthesis_queue_timeout`; `0` disables waiting. |

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

Preloading is recommended for unattended processes because it moves missing
dependency, missing artifact, and invalid config failures to startup. Lazy
loading is useful during development, but the first speech request will pay the
load cost and may surface setup errors to the client.

## Logs and Common Failures

Uvicorn writes access logs and application errors to stdout/stderr. Under
launchd, redirect them to files under a writable log directory and rotate them
with your normal host policy. Avoid logging bearer tokens, request bodies that
contain private text, or full local filesystem paths in shared channels.

Common startup and runtime failures:

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `/health` shows `runtime: unconfigured` | No weights source was set. | Configure one supported weight source from [real_model_setup.md](real_model_setup.md). |
| `/health` shows `server.status: "configuration_error"` | Server env parsing failed, commonly an invalid concurrency or queue timeout value. | Check `server.error.message`; fix integer, boolean, or missing server config values. |
| `/v1/*` returns `server_configuration_error` | Server env parsing failed, commonly an invalid concurrency or queue timeout value. | Fix `IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS` or `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS`, then restart. |
| `/v1/*` returns `invalid_api_key` | Bearer token is enabled but missing or wrong. | Send `Authorization: Bearer <token>` and verify the active token source. |
| `/v1/audio/speech` returns `runtime_unavailable` | Irodori-TTS-MLX deps, weights, or model config failed at load time. | Run with `IRODORI_MLX_PRELOAD=1`, inspect logs, and rerun the real setup smoke. |
| `/v1/audio/speech` returns `synthesis_queue_timeout` | All synthesis slots are busy. | Increase client backoff or queue timeout before raising concurrency. |
| Process exits immediately under launchd | Bad working directory, venv path, or env file. | Run the launch command manually as the same user, then check launchd stderr. |

## launchd Example

Use a plist when the server should start at login for one macOS user. Adjust
paths, model source, and token loading for your host; keep secrets out of the
repository.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.irodori.tts-mlx-server</string>
  <key>WorkingDirectory</key>
  <string>/opt/irodori-tts-mlx-server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/irodori-tts-mlx-server/.venv/bin/python</string>
    <string>-m</string>
    <string>irodori_tts_mlx_server</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8000</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>IRODORI_MLX_WEIGHTS_DIR</key>
    <string>/opt/irodori-models/voicedesign-v2</string>
    <key>IRODORI_MLX_PRELOAD</key>
    <string>1</string>
    <key>IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS</key>
    <string>1</string>
    <key>IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS</key>
    <string>60</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/opt/irodori-tts-mlx-server/logs/server.log</string>
  <key>StandardErrorPath</key>
  <string>/opt/irodori-tts-mlx-server/logs/server.err.log</string>
</dict>
</plist>
```

Load it after validating the command manually:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.irodori.tts-mlx-server.plist
launchctl kickstart -k gui/$(id -u)/dev.irodori.tts-mlx-server
curl --fail http://127.0.0.1:8000/health
```

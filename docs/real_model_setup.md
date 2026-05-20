# Real Model Setup and Converted Weights

This page describes the supported ways to start this server with real
Irodori-TTS-MLX weights from a fresh checkout. The repository does not include
model weights, tokenizer assets, upstream source code, reference audio, generated
audio, secrets, or Hugging Face cache snapshots.

The server runtime boundary is the `irodori_mlx.runtime` adapter from
[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX). Install this
server and install Irodori-TTS-MLX runtime dependencies into the same Python
environment before using any real model configuration:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

# Install Irodori-TTS-MLX from a local checkout or a published package source.
# The runtime must provide the irodori_mlx package and its DACVAE bridge deps.
python -m pip install -e /path/to/Irodori-TTS-MLX
```

If Irodori-TTS-MLX cannot import its runtime dependencies, `/v1/audio/speech`
returns a 503 OpenAI-style `runtime_unavailable` error instead of failing at
server import time.

## Supported Weight Sources

Use exactly one of these source modes.

| Source | CLI flag | Environment variable | Use when |
| --- | --- | --- | --- |
| Hosted converted weights repo | `--weights-repo owner/repo` | `IRODORI_MLX_WEIGHTS_REPO=owner/repo` | A Hugging Face repository already follows the converted-weights layout and is approved for your use. |
| Local converted-weights layout directory | `--weights-dir /models/irodori-layout` | `IRODORI_MLX_WEIGHTS_DIR=/models/irodori-layout` | You have the same hosted layout on disk, commonly for private staging or offline use. |

`IRODORI_MLX_WEIGHTS_REVISION` or `--weights-revision` may be supplied with a
hosted repository to pin a branch, tag, or commit. `IRODORI_MLX_PRELOAD=1` or
`--preload` loads the runtime during server startup; otherwise the first speech
request loads it lazily.

## Converted Weights Layout

`IRODORI_MLX_WEIGHTS_REPO` and `IRODORI_MLX_WEIGHTS_DIR` expect the
Irodori-TTS-MLX hosted weights layout at the repository or directory root:

```text
repo-or-local-dir/
+-- README.md
+-- LICENSE.md
+-- irodori_mlx_manifest.json
+-- model_config.json
+-- tokenizer_config.json
+-- conversion_metadata.json
+-- weights.npz
+-- checksums.sha256
```

The loader treats `irodori_mlx_manifest.json` as the source of truth. At minimum,
the layout should identify the converted family, upstream checkpoint, converter
version, required runtime version, file names, checksum coverage, and license
review status. Public hosted repositories should use only artifacts whose
license review is approved for redistribution.

`weights.npz` is the converted MLX RF-DiT weights archive produced by the
Irodori-TTS-MLX converter. It must not contain original upstream safetensors,
Hugging Face cache snapshots, DACVAE weights, reference audio, generated audio,
or unrelated model artifacts.

`model_config.json` must be accepted by `irodori_mlx.config.ModelConfig` for the
converted family. For VoiceDesign v2 caption-conditioned weights, it should
enable caption conditioning. For v3 weights, it should include the validated
duration-predictor configuration. `tokenizer_config.json` and
`conversion_metadata.json` should preserve the tokenizer/conditioning contract
and conversion provenance.

## Minimal Local Run

Hosted or local-layout source:

```bash
IRODORI_MLX_WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
IRODORI_MLX_WEIGHTS_REVISION=main \
python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000
```

Check runtime status without forcing a load:

```bash
curl http://127.0.0.1:8000/health
```

A configured but not yet loaded runtime reports `configured: true`,
`loaded: false`, and a `weights_source` such as `weights_repo` or `weights_dir`.
If `IRODORI_MLX_PRELOAD=1` is set, startup attempts the model
load immediately and reports any load error through `/health`.

## Minimal Speech Request

Once the server is running with a configured runtime, request a WAV response:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "irodori-tts-mlx",
    "input": "Hello from Irodori TTS MLX.",
    "voice": "voicedesign",
    "response_format": "wav",
    "irodori": {
      "no_ref": true,
      "caption": "calm narration, clear diction",
      "preset": "balanced"
    }
  }' \
  --output speech.wav
```

If `IRODORI_SERVER_BEARER_TOKEN` or `IRODORI_API_KEY` is configured, add:

```bash
-H 'Authorization: Bearer <token>'
```

The command should create a non-empty `speech.wav` file. For the current
VoiceDesign no-reference path, set `irodori.no_ref=true` and provide a
caption. Do not set `irodori.ref_wav` together with
`irodori.no_ref=true`.

## Common Configuration Failures

All API errors use the OpenAI-compatible JSON shape:

```json
{
  "error": {
    "message": "...",
    "type": "server_error",
    "param": null,
    "code": "runtime_unavailable"
  }
}
```

Common setup failures:

| Symptom | Status | Error code | Expected fix |
| --- | --- | --- | --- |
| No weights source is configured. | 503 | `runtime_unavailable` | Set `IRODORI_MLX_WEIGHTS_REPO` or `IRODORI_MLX_WEIGHTS_DIR`. |
| Irodori-TTS-MLX runtime imports fail. | 503 | `runtime_unavailable` | Install Irodori-TTS-MLX and its runtime dependencies in the server venv. |
| Hosted repo or local layout is missing required files or fails validation. | 503 | `runtime_unavailable` | Use a layout with manifest, config, tokenizer metadata, conversion metadata, weights, and checksums matching the Irodori-TTS-MLX layout contract. |
| `IRODORI_MLX_MAX_TEXT_LEN` or `IRODORI_MLX_MAX_CAPTION_LEN` is not an integer. | 503 | `runtime_unavailable` | Use an integer value or unset it. |
| Server settings such as `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS` are invalid. | 503 | `server_configuration_error` | Use finite, non-negative queue timeout and a concurrency value of at least 1. |
| Bearer auth is enabled and the request omits or uses the wrong token. | 401 | `invalid_api_key` | Send `Authorization: Bearer <token>`. |
| The request names a model id not listed by `/v1/models`. | 404 | `model_not_found` | Use `irodori-tts-mlx` unless `IRODORI_MLX_MODEL_ID` intentionally changes it. |
| VoiceDesign options conflict, for example `ref_wav` with `no_ref=true`. | 400 | `invalid_irodori_options` | Use either no-reference caption generation or reference-audio generation, not both. |

Do not paste tokens into shell history or commit `.env` files. Keep converted
weights and generated audio outside the repository unless a separate release
process explicitly approves publishing them.

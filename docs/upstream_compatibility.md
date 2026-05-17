# Upstream Compatibility Gap Matrix

This page compares this MLX-focused server with
[Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).
The goal is not full parity by default. It is to make the current compatibility
contract explicit so future work can prioritize the gaps that matter.

Status values:

- Implemented: available in this repository.
- Partial: available with a narrower behavior, different option name, or a
  known setup caveat.
- Unsupported: not available today and a candidate for future work.
- Intentionally out of scope: not planned for this MLX server unless the project
  direction changes.

## Matrix

| Area | Upstream Irodori-TTS-Server | This MLX server | Status | Gap / rationale |
| --- | --- | --- | --- | --- |
| OpenAI-compatible model list | `GET /v1/models` returns the configured `irodori-tts` model id. | `GET /v1/models` returns `irodori-tts-mlx` by default, or `IRODORI_MLX_MODEL_ID` when overridden. | Implemented | The model id differs because this server targets converted MLX weights rather than the upstream PyTorch checkpoint. |
| Speech endpoint route | `POST /v1/audio/speech` returns a complete audio response. | `POST /v1/audio/speech` returns a complete audio response. | Implemented | Streaming synthesis is not supported by either server. |
| Speech request base fields | Accepts `model`, `input`, `voice`, `response_format`, `speed`, and `irodori`. Upstream also accepts extra compatibility fields. | Accepts `model`, `input`, `voice`, `response_format`, `speed`, `stream`, and `irodori`; unknown top-level fields are rejected. | Partial | This server keeps the OpenAI-compatible request surface strict. Add specific top-level aliases only when a real client needs them. |
| Model id default | Defaults to `irodori-tts`. | Defaults to `irodori-tts-mlx`. | Partial | Use this server model id in clients, or set `IRODORI_MLX_MODEL_ID` if a local compatibility alias is required. |
| Response formats | Supports `wav`, `mp3`, `flac`, `opus`, `aac`, and `pcm`; FFmpeg is needed for encoded formats. | Supports `wav` and `pcm` without optional encoders, and `mp3`, `flac`, `opus`, and `aac` through FFmpeg. OpenAI default `mp3` is preserved. | Implemented | The runtime always produces WAV first, then this server converts the response when needed. |
| Streaming response / SSE | Rejects `stream_format: "sse"` because generation is non-streaming. | Rejects `stream=true` and SSE `Accept` requests with `unsupported_streaming`. | Implemented | Client SDK helpers named `with_streaming_response` can still download the complete response. |
| Reference voice support | Resolves voices from files, `voices.json`, HTTP uploads, or a voice object. | Per-request `irodori.reference_wav` can be passed to the MLX runtime when `irodori.no_reference=false`. Managed `.wav` files uploaded under `/v1/audio/voices` can also be selected by `voice` id. | Partial | The MLX server intentionally keeps reference management narrow: no alias file, latent reference, remote URL, non-WAV upload, or voice object normalization yet. |
| VoiceDesign caption / no-reference support | Supports upstream no-reference behavior through the PyTorch runtime and voice registry conventions. | Supports VoiceDesign v2 no-reference caption generation with `irodori.no_reference=true` and `irodori.caption`. Hosted converted VoiceDesign weights are documented in `docs/real_model_setup.md`. | Partial | This is the primary MLX path, but real smoke validation remains opt-in and depends on converted weights plus the external Irodori-TTS-MLX runtime package. |
| Irodori runtime options | Supports common v3 sampling controls such as `num_steps`, `cfg_scale_text`, `cfg_scale_speaker`, `seed`, schedule controls, and per-request `lora_adapter`. | Supports `preset`, `seconds`, `duration_scale`, `num_steps`, `seed`, `cfg_scale_text`, `cfg_scale_caption`, `cfg_scale_speaker`, `cfg_guidance_mode`, `cfg_min_t`, `cfg_max_t`, `max_reference_seconds`, and context-cache control. | Partial | Option names follow the MLX runtime adapter. Dynamic LoRA and upstream schedule-specific fields are not implemented. |
| Long-text chunking | Automatically chunks long text, with request/default controls such as `chunking_enabled` and `chunk_min_chars`; skips chunking when explicit seconds are set. | Chunks by punctuation and hard character slices by default using `irodori.chunking` and `irodori.chunk_max_chars`; explicit `irodori.seconds` is distributed across chunks by character count. | Partial | The behavior is implemented but not identical. The MLX server favors max-size splitting and duration distribution. |
| Queue / concurrency controls | Serializes expensive synthesis work with queue timeout controls. | Uses a synthesis semaphore controlled by `IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS` and `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS`; queue exhaustion returns `synthesis_queue_timeout`. | Implemented | Defaults are local-machine oriented: one synthesis at a time and a 30-second queue timeout. |
| Bearer-token auth | Optional bearer-token auth on OpenAI-compatible routes. | Optional bearer-token auth on `/v1/*` routes via `IRODORI_SERVER_BEARER_TOKEN`; `IRODORI_API_KEY` is accepted as an alias. `/health` stays unauthenticated. | Implemented | This matches the intended local/OpenAI-compatible auth shape. |
| Docker / deployment support | Includes `Dockerfile`, Compose files, GPU-oriented NVIDIA/CUDA guidance, mounted voices, and Hugging Face cache volume reuse. | Local Python and Apple Silicon setup are documented. No Dockerfile, Compose file, launchd service, or production reverse-proxy guide exists yet. | Unsupported | Docker is less central for MLX/Apple Silicon, but a local-host deployment guide is still tracked by the post-MVP epic. |
| Voice management endpoints | Provides `GET`, `POST`, `GET by id`, `PUT`, and `DELETE` routes under `/v1/audio/voices`. | Provides the same route shape for managed `.wav` reference files under `IRODORI_SERVER_VOICES_DIR`. | Partial | This is a minimal useful subset for OpenAI-style clients. It omits upstream alias-file scanning, latent `.pt`/`.pth` references, non-WAV upload formats, and broad voice-object normalization because those would expand the MLX server's storage and runtime contract. |
| Runtime / model backend | Uses the PyTorch Irodori-TTS 500M v3 base model from Hugging Face or a local safetensors checkpoint, with CUDA-oriented deployment. | Uses the external `irodori_mlx.runtime` adapter with hosted or local converted MLX weights: `IRODORI_MLX_WEIGHTS_REPO`, `IRODORI_MLX_WEIGHTS_DIR`, or `IRODORI_MLX_WEIGHTS_PATH` plus `IRODORI_MLX_MODEL_CONFIG_JSON`. | Partial | The backend is intentionally different. Fresh checkout setup and converted-weight layout are documented, but real-weight smoke testing is opt-in and may take several minutes. |
| CUDA / PyTorch checkpoint serving | Provides the upstream CUDA/PyTorch server path for `Aratako/Irodori-TTS-500M-v3`. | Does not serve PyTorch safetensors checkpoints or CUDA runtimes. | Intentionally out of scope | This repository exists to expose Irodori-TTS-MLX on Apple Silicon. Use the upstream server for the PyTorch/CUDA stack. |
| Dynamic LoRA adapters | Supports per-request `irodori.lora_adapter` with runtime caching, except when model compilation is enabled. | Not implemented. | Unsupported | The current MLX adapter boundary does not expose dynamic LoRA loading. |
| Health endpoint | Reports model, runtime, voices, defaults, and queue configuration without loading the model. | Reports runtime configuration/load state and server auth/concurrency settings without forcing model load. | Partial | Voice-registry and PyTorch-specific fields do not exist in this server. |
| Error shape | Uses OpenAI-style JSON errors for validation, auth, runtime, and queue failures. | Uses OpenAI-style JSON errors for validation, auth, runtime, encoder, and queue failures. | Implemented | Error codes differ where backend-specific failures differ. |

## Managed reference voice scope

Issue #24 evaluated the upstream voice/reference API and implemented the
smallest scope that is useful for this MLX server: managed `.wav` reference file
CRUD at `/v1/audio/voices` plus speech-time resolution from `voice="<id>"` to
`irodori.reference_wav`. The implementation deliberately avoids upstream
features that can read or normalize references outside the managed directory:
`voices.json` aliases, absolute or relative path aliases, latent `.pt`/`.pth`
files, remote URLs, and non-WAV uploads are deferred.

Storage and security assumptions:

- `IRODORI_SERVER_VOICES_DIR` chooses the managed storage root and defaults to
  `voices` relative to the server process.
- `voice_id` accepts only ASCII letters, numbers, underscores, and hyphens.
- Uploaded files are written as `<voice_id>.wav` inside the managed directory.
- A speech request using a managed `voice` id only injects the managed file path
  when the request did not already provide explicit `irodori.reference_wav` or
  `irodori.no_reference` options.
- Direct `irodori.reference_wav` remains an explicit local-server option for
  trusted deployments; the management API itself does not resolve arbitrary
  client-supplied paths.

## Current Priorities

The post-MVP readiness work should treat these as the highest-value gaps:

1. Complete the opt-in real MLX runtime smoke path with converted weights and
   record non-empty WAV/PCM evidence.
2. Decide whether persistent voice/reference management belongs in this server
   or remains an upstream-only feature.
3. Add local Apple Silicon deployment guidance for a long-running server process.
4. Add OpenAI-compatible client examples for the default MLX model id and
   VoiceDesign caption path.

## Runtime Validation Context

The documented real-model setup currently points at the hosted converted
VoiceDesign weights source `t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign` and
keeps real MLX smoke tests opt-in so normal CI does not download large artifacts.
That setup is compatibility context, not a parity claim: until the opt-in smoke
test records a completed WAV/non-empty PCM run on a machine with the external
Irodori-TTS-MLX runtime and converted weights installed, this matrix should keep
runtime parity marked as partial.

## Compatibility Regression Tests

Run the deterministic OpenAI/upstream compatibility regression suite locally
with:

```bash
python -m pytest tests/test_compatibility.py
```

These tests use a fake runtime rather than real MLX weights. They cover
representative OpenAI Python client calls for model listing, bearer auth,
non-streaming speech download, unsupported streaming, and response format
conversion, plus upstream-style request fixtures for supported options and
stable OpenAI-style error codes for unsupported behavior.

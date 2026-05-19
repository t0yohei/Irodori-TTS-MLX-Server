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
| Speech endpoint route | `POST /v1/audio/speech` returns a complete audio response. | `POST /v1/audio/speech` returns a complete audio response. The additive `POST /v1/audio/speech/stream-chunks` extension returns chunk-level SSE events. | Implemented | The OpenAI-compatible route stays non-streaming; chunk-level SSE is an MLX-server extension for lower perceived latency. |
| Speech request base fields | Accepts `model`, `input`, `voice`, `response_format`, `speed`, `stream_format`, and `irodori`. Upstream also accepts extra compatibility fields. | Accepts `model`, `input`, `voice`, `response_format`, `speed`, `stream`, `stream_format`, and `irodori`. Unknown top-level fields are rejected except for explicit runtime option aliases that normalize into `irodori`. | Partial | The accepted aliases are narrow and contract-tested; path-style reference aliases and unknown top-level fields still fail validation. |
| Model id default | Defaults to `irodori-tts`. | Defaults to `irodori-tts-mlx`. | Partial | Use this server model id in clients, or set `IRODORI_MLX_MODEL_ID` if a local compatibility alias is required. |
| Response formats | Supports `wav`, `mp3`, `flac`, `opus`, `aac`, and `pcm`; FFmpeg is needed for encoded formats. | Supports `wav` and `pcm` without optional encoders, and `mp3`, `flac`, `opus`, and `aac` through FFmpeg. OpenAI default `mp3` is preserved. | Implemented | The runtime always produces WAV first, then this server converts the response when needed. |
| Streaming response / SSE | Rejects `stream_format: "sse"` because generation is non-streaming. | Rejects `stream=true`, any `stream_format`, and SSE `Accept` requests on `/v1/audio/speech` with `unsupported_streaming`; supports chunk-level SSE on `/v1/audio/speech/stream-chunks`. | Partial | Client SDK helpers named `with_streaming_response` can still download the complete response. The chunk SSE route is not part of the OpenAI-compatible speech contract. |
| Reference voice support | Resolves voices from files, `voices.json`, HTTP uploads, or a voice object. | Managed audio files uploaded under `/v1/audio/voices` can be selected by `voice` id or `{"id":"voice_id"}`; explicit `irodori.reference_wav` must resolve to an existing managed file under `IRODORI_SERVER_VOICES_DIR`. | Partial | The MLX server accepts common managed audio extensions but intentionally rejects alias files, latent references, remote URLs, path traversal, and arbitrary local paths. |
| VoiceDesign caption / no-reference support | Supports upstream no-reference behavior through the PyTorch runtime and voice registry conventions. | Supports VoiceDesign v2 no-reference caption generation with `irodori.no_reference=true` and `irodori.caption`. Hosted converted VoiceDesign weights are documented in `docs/real_model_setup.md`. | Partial | This is the primary MLX path, but real smoke validation remains opt-in and depends on converted weights plus the external Irodori-TTS-MLX runtime package. |
| Irodori runtime options | Supports common v3 sampling controls such as `num_steps`, `cfg_scale_text`, `cfg_scale_speaker`, `seed`, schedule controls, and per-request `lora_adapter`. | Supports `preset` (`ultra-fast`, `fast`, `balanced`, `quality`), `seconds`, `duration_scale`, `num_steps`, `seed`, `cfg_scale_text`, `cfg_scale_caption`, `cfg_scale_speaker`, `cfg_guidance_mode`, `cfg_min_t`, `cfg_max_t`, `max_reference_seconds`, and context-cache control. Upstream aliases `no_ref`, `ref_wav`, `max_ref_seconds`, `context_kv_cache`, and `chunking_enabled` are normalized when unambiguous. `irodori.lora_adapter` is parsed but rejected. | Partial | Dynamic LoRA, latent references, `chunk_min_chars`, and PyTorch schedule-specific fields are explicitly rejected rather than silently ignored. |
| Long-text chunking | Automatically chunks long text, with request/default controls such as `chunking_enabled` and `chunk_min_chars`; skips chunking when explicit seconds are set. | Chunks by punctuation and hard character slices by default using `irodori.chunking` and `irodori.chunk_max_chars`; accepts `chunking_enabled` as an alias for enable/disable; explicit `irodori.seconds` is distributed across chunks by character count. | Partial | `chunk_min_chars` is intentionally rejected because upstream minimum-split semantics do not match the MLX server's maximum-size splitter. |
| Queue / concurrency controls | Serializes expensive synthesis work with queue timeout controls. | Uses a synthesis semaphore controlled by `IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS` and `IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS`; queue exhaustion returns `synthesis_queue_timeout`. | Implemented | Defaults are local-machine oriented: one synthesis at a time and a 30-second queue timeout. |
| Bearer-token auth | Optional bearer-token auth on OpenAI-compatible routes. | Optional bearer-token auth on `/v1/*` routes via `IRODORI_SERVER_BEARER_TOKEN`; `IRODORI_API_KEY` is accepted as an alias. `/health` stays unauthenticated. | Implemented | This matches the intended local/OpenAI-compatible auth shape. |
| Docker / deployment support | Includes `Dockerfile`, Compose files, GPU-oriented NVIDIA/CUDA guidance, mounted voices, and Hugging Face cache volume reuse. | Local Python and Apple Silicon setup are documented, with checked-in launchd and environment templates for local packaged deployment. Docker and Compose are not provided. | Partial | The supported packaged target is macOS launchd for Apple Silicon MLX. Docker/Compose remains out of the default path because this repo should not imply CUDA or PyTorch container support. |
| Voice management endpoints | Provides `GET`, `POST`, `GET by id`, `PUT`, and `DELETE` routes under `/v1/audio/voices`. | Provides the same route shape for managed audio reference files under `IRODORI_SERVER_VOICES_DIR`, including `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.aac`, and `.webm`. | Partial | This is a safe subset for OpenAI-style clients. It omits upstream alias-file scanning, latent `.pt`/`.pth` references, and broad voice-object normalization because those would expand the MLX server's storage and runtime contract. |
| Runtime / model backend | Uses the PyTorch Irodori-TTS 500M v3 base model from Hugging Face or a local safetensors checkpoint, with CUDA-oriented deployment. | Uses the external `irodori_mlx.runtime` adapter with hosted or local converted MLX weights: `IRODORI_MLX_WEIGHTS_REPO` or `IRODORI_MLX_WEIGHTS_DIR`. | Partial | The backend is intentionally different. Fresh checkout setup and converted-weight layout are documented, but real-weight smoke testing is opt-in and may take several minutes. |
| CUDA / PyTorch checkpoint serving | Provides the upstream CUDA/PyTorch server path for `Aratako/Irodori-TTS-500M-v3`. | Does not serve PyTorch safetensors checkpoints or CUDA runtimes. | Intentionally out of scope | This repository exists to expose Irodori-TTS-MLX on Apple Silicon. Use the upstream server for the PyTorch/CUDA stack. |
| Dynamic LoRA adapters | Supports per-request `irodori.lora_adapter` with runtime caching, except when model compilation is enabled. | Not implemented. Requests with a non-empty `irodori.lora_adapter` return `invalid_irodori_options`; path-like values are rejected explicitly. | Unsupported | The current Irodori-TTS-MLX `GenerationRequest` and `MLXDACVAERuntime` boundary does not expose dynamic LoRA loading or adapter cache/reload semantics. This server therefore does not add PyTorch/CUDA assumptions or arbitrary local path loading. |
| Health endpoint | Reports model, runtime, voices, defaults, and queue configuration without loading the model. | Reports runtime configuration and explicit `load_state` values (`unconfigured`, `not_loaded`, `loading`, `loaded`, `failed`) plus server auth/concurrency settings without forcing model load. | Partial | Voice-registry and PyTorch-specific fields do not exist in this server. |
| Error shape | Uses OpenAI-style JSON errors for validation, auth, runtime, and queue failures. | Uses OpenAI-style JSON errors for validation, auth, runtime, encoder, and queue failures. | Implemented | Error codes differ where backend-specific failures differ. |

## Managed reference voice scope

Issue #38 re-audited the upstream voice/reference API after the initial #24
managed-WAV subset. The safe MLX-compatible scope is managed audio file CRUD at
`/v1/audio/voices`, speech-time resolution from `voice="<id>"` or
`voice={"id":"<id>"}` to `irodori.reference_wav`, and explicit
`irodori.reference_wav` only when the path resolves to an existing managed file.
The implementation deliberately avoids upstream features that can read or
normalize references outside the managed directory: `voices.json` aliases,
absolute or relative path aliases that escape the managed root, latent
`.pt`/`.pth` files, remote URLs, and symbolic links are rejected or ignored.

Storage and security assumptions:

- `IRODORI_SERVER_VOICES_DIR` chooses the managed storage root and defaults to
  `voices` relative to the server process.
- `voice_id` accepts only ASCII letters, numbers, underscores, and hyphens.
- Uploaded files are written as `<voice_id><extension>` inside the managed
  directory. Supported extensions are `.wav`, `.flac`, `.mp3`, `.m4a`,
  `.ogg`, `.opus`, `.aac`, and `.webm`.
- New uploads and replacements are committed atomically so interrupted writes do
  not expose partial managed voice files.
- A speech request using a managed `voice` id only injects the managed file path
  when the request did not already provide explicit `irodori.reference_wav` or
  a no-reference option other than false. Upstream-style `no_ref=false` keeps
  managed voice resolution enabled.
- Direct `irodori.reference_wav` is no longer an arbitrary local-server path:
  it must be a managed, non-symlink file under `IRODORI_SERVER_VOICES_DIR`.

Audit classification:

- Implemented: managed common audio upload extensions, `voice={"id":"..."}`
  normalization, and managed-root validation for explicit `irodori.reference_wav`.
- Deferred: conversion of uploaded non-WAV files to WAV. The MLX runtime adapter
  receives the managed audio path and remains the decode boundary; adding a
  server-side transcoder would introduce new optional audio tooling.
- Out of scope for this Apple Silicon MLX server: arbitrary local path aliases,
  remote URL references, symlinked managed files, and upstream latent
  `.pt`/`.pth` references.

## Dynamic LoRA adapter status

Issue #37 evaluated the current Irodori-TTS-MLX runtime API. The MLX runtime
request contract exposes sampling, speaker/reference, caption, duration, and
context-cache controls, but it does not expose a clean LoRA adapter field,
adapter allowlist, cache key, reload hook, or unload behavior. This server keeps
LoRA unsupported until that contract exists in the MLX runtime.

Runtime behavior:

- Omitted or empty `irodori.lora_adapter` values are ignored for compatibility
  with clients that send blank optional fields.
- Non-string `irodori.lora_adapter` values return `invalid_irodori_options`.
- Path-like values such as `../adapter.safetensors`, `/models/a.safetensors`,
  `~/adapter`, or drive-style strings are rejected before runtime generation.
- Safe-looking aliases such as `warm-narration` also return
  `invalid_irodori_options` with an unsupported-runtime message.

Future MLX support should require an explicit configured alias or allowlist
root, should not load arbitrary request paths, and should document cache/reload
semantics before enabling per-request adapter selection.

## Current Priorities

The post-MVP readiness work should treat these as the highest-value gaps:

1. Complete the opt-in real MLX runtime smoke path with converted weights and
   record non-empty WAV/PCM evidence.
2. Decide whether managed voice/reference storage belongs in this server
   or remains an upstream-only feature.
3. Broaden compatibility regression coverage for representative OpenAI and
   upstream-style clients.
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

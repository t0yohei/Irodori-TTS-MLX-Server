# OpenAI-Compatible Client Examples

These examples call the local Irodori-TTS-MLX server through the OpenAI-style
`/v1/models` and `/v1/audio/speech` routes. They assume the server is already
running on `http://127.0.0.1:8000` with a configured runtime, as described in
[real_model_setup.md](real_model_setup.md).

If bearer auth is enabled with `IRODORI_SERVER_BEARER_TOKEN` or `IRODORI_API_KEY`,
send the same value as `Authorization: Bearer <token>`. If auth is disabled, the
header is ignored by the server; OpenAI SDKs may still require a placeholder API
key in client configuration.

## curl

List available models:

~~~bash
curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer <token>'
~~~

Generate WAV audio without optional encoders:

~~~bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{
    "model": "irodori-tts-mlx",
    "input": "Hello from the OpenAI-compatible curl client.",
    "voice": "voicedesign",
    "response_format": "wav",
    "irodori": {
      "no_ref": true,
      "caption": "calm narration, clear diction",
      "preset": "balanced",
      "t_schedule_mode": "sway",
      "sway_coeff": -1.0
    }
  }' \
  --output speech.wav
~~~

Generate a converted response format when FFmpeg is installed:

~~~bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{
    "model": "irodori-tts-mlx",
    "input": "This response is encoded as MP3 by FFmpeg.",
    "voice": "voicedesign",
    "response_format": "mp3",
    "irodori": {
      "no_ref": true,
      "caption": "bright young voice, energetic delivery",
      "preset": "fast"
    }
  }' \
  --output speech.mp3
~~~

`mp3`, `flac`, `opus`, and `aac` require FFmpeg on the server host. Without
FFmpeg, request `wav` or `pcm`; the server returns an OpenAI-style
`response_format_unavailable` error for encoded formats.

## Python OpenAI Client

Install the OpenAI Python client in your application environment:

~~~bash
python -m pip install openai
~~~

Configure `base_url` to point at the server's `/v1` prefix. The SDK sends the
configured `api_key` as a bearer token.

~~~python
from pathlib import Path
import os

from openai import OpenAI


client = OpenAI(
    base_url=os.environ.get("IRODORI_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=(
        os.environ.get("IRODORI_SERVER_BEARER_TOKEN")
        or os.environ.get("IRODORI_API_KEY")
        or "local-dev-token"
    ),
)

models = client.models.list()
print([model.id for model in models.data])

speech = client.audio.speech.create(
    model="irodori-tts-mlx",
    voice="voicedesign",
    input="Hello from the OpenAI Python client.",
    response_format="wav",
    extra_body={
        "irodori": {
            "no_ref": True,
            "caption": "calm narration, clear diction",
            "preset": "balanced",
            "rescale_k": 0.7,
            "rescale_sigma": 1.2,
        }
    },
)
speech.write_to_file(Path("speech.wav"))
~~~

For FFmpeg-backed formats, change `response_format` and the output path:

~~~python
speech = client.audio.speech.create(
    model="irodori-tts-mlx",
    voice="voicedesign",
    input="This response is encoded as FLAC by FFmpeg.",
    response_format="flac",
    extra_body={
        "irodori": {
            "no_ref": True,
            "caption": "clear studio narration",
            "preset": "balanced",
        }
    },
)
speech.write_to_file(Path("speech.flac"))
~~~

The OpenAI Python client does not need server-specific transport code. Use
`extra_body` for Irodori-specific options such as `no_ref`, `ref_embed`,
`caption`, `preset`, `seed`, `num_steps`, schedule/quality knobs, or
reference audio paths. `ref_embed` paths must point at managed
`.speaker.safetensors` files under the server voices directory.

## Streaming

The OpenAI-compatible `/v1/audio/speech` route supports OpenAI-style streaming.
Use `stream_format="sse"` or `Accept: text/event-stream` to receive
Server-Sent Events from the same route. `stream_format="audio"` is not
supported.

SSE streaming emits one audio delta per synthesized text chunk:

~~~bash
curl -N http://127.0.0.1:8000/v1/audio/speech \\
  -H 'Content-Type: application/json' \\
  -H 'Accept: text/event-stream' \\
  -H 'Authorization: Bearer <token>' \\
  -d '{
    "model": "irodori-tts-mlx",
    "input": "最初の文です。次の文です。",
    "voice": "voicedesign",
    "response_format": "wav",
    "stream_format": "sse",
    "irodori": {
      "no_ref": true,
      "caption": "clear studio narration",
      "chunking_enabled": true,
      "chunk_min_chars": 80,
      "first_sentence_chunk_min_chars": 1
    }
  }'
~~~

Chunking uses punctuation boundaries by default after `chunk_min_chars`
non-space characters. Use `first_sentence_chunk_min_chars` only when you want
the first audio event as early as possible. It applies a separate threshold to
the first sentence; later sentences keep the normal punctuation chunk planner.

The response uses Server-Sent Events:

~~~text
event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"..."}

event: speech.audio.done
data: {"type":"speech.audio.done","usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}
~~~

`audio` contains base64-encoded audio bytes for the completed text chunk.
Clients can decode and enqueue each `speech.audio.delta` while the server generates
later chunks. The local server does not perform OpenAI token accounting, so the
terminal event includes a zero-valued `usage` object for schema compatibility.

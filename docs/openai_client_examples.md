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
      "preset": "balanced"
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
`extra_body` for Irodori-specific options such as `no_ref`, `caption`,
`preset`, `seed`, `num_steps`, or reference audio paths.

## Streaming

The OpenAI-compatible `/v1/audio/speech` route supports OpenAI-style streaming.
Use `stream_format="audio"` to receive audio bytes on the response body, or
`stream_format="sse"` to receive Server-Sent Events from the same route.

Audio byte streaming is useful for clients that can play the response body
directly:

~~~bash
curl http://127.0.0.1:8000/v1/audio/speech \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer <token>' \\
  -d '{
    "model": "irodori-tts-mlx",
    "input": "This response streams audio bytes.",
    "voice": "voicedesign",
    "response_format": "wav",
    "stream_format": "audio",
    "irodori": {
      "no_ref": true,
      "caption": "clear studio narration"
    }
  }' | ffplay -i -
~~~

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
      "punctuation_chunking_enabled": true,
      "first_sentence_comma_chunking_enabled": true
    }
  }'
~~~

Use `first_sentence_comma_chunking_enabled` only with punctuation chunking when
you want the first audio event as early as possible. It splits only the first
sentence on comma-like punctuation such as `、`; later sentences keep the normal
punctuation chunk planner.

The response uses Server-Sent Events:

~~~text
event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"..."}

event: speech.audio.done
data: {"type":"speech.audio.done"}
~~~

`audio` contains base64-encoded audio bytes for the completed text chunk.
Clients can decode and enqueue each `speech.audio.delta` while the server generates
later chunks.

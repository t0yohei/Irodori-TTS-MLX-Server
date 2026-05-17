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
      "no_reference": true,
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
      "no_reference": true,
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
            "no_reference": True,
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
            "no_reference": True,
            "caption": "clear studio narration",
            "preset": "balanced",
        }
    },
)
speech.write_to_file(Path("speech.flac"))
~~~

The OpenAI Python client does not need server-specific transport code. Use
`extra_body` for Irodori-specific options such as `no_reference`, `caption`,
`preset`, `seed`, `num_steps`, or reference audio paths.

## Streaming

Streaming synthesis and SSE responses are not supported by this server. Do not
send `stream=true` or `Accept: text/event-stream` for `/v1/audio/speech`.
Those requests return a 400 OpenAI-style error with
`code="unsupported_streaming"`. Request a complete audio file response instead.
SDK helpers that stream the HTTP download body are different from synthesis
streaming; they still receive one completed audio file from the server.

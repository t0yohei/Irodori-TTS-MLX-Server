#!/usr/bin/env python3
"""Stream Irodori speech SSE chunks and autoplay them with ffplay.

Expected server setup:

    IRODORI_MLX_WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v3
    IRODORI_SERVER_VOICES_DIR=/Users/kouka/.openclaw/workspace/repos/Irodori-TTS-MLX-Server/voices

The default request uses voices/stream-fast-wide-reference.wav as a managed
reference voice through irodori.ref_wav.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from typing import Any


DEFAULT_TEXT = (
    "こんにちは。これは v3 の reference voice を使ったストリーミング再生テストです。"
    "最初の音声を再生しながら、次の chunk の生成を待ちます。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5058/v1/audio/speech")
    parser.add_argument("--model", default="irodori-tts-mlx")
    parser.add_argument("--voice", default="stream-fast-wide-reference")
    parser.add_argument("--ref-wav", default="stream-fast-wide-reference.wav")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--preset", default="ultra-fast")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "mp3", "flac", "opus", "aac"],
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chunk-min-chars", type=int)
    parser.add_argument("--bearer-token")
    parser.add_argument("--save-prefix", help="Optional path prefix for received chunks.")
    parser.add_argument("--no-play", action="store_true", help="Receive chunks without ffplay playback.")
    return parser.parse_args()


def request_payload(args: argparse.Namespace) -> dict[str, Any]:
    irodori: dict[str, Any] = {
        "ref_wav": args.ref_wav,
        "no_ref": False,
        "preset": args.preset,
        "chunking_enabled": True,
        "punctuation_chunking_enabled": True,
        "first_sentence_comma_chunking_enabled": True,
    }
    if args.num_steps is not None:
        irodori["num_steps"] = args.num_steps
    if args.seed is not None:
        irodori["seed"] = args.seed
    if args.chunk_min_chars is not None:
        irodori["chunk_min_chars"] = args.chunk_min_chars
    return {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
        "stream_format": "sse",
        "speed": args.speed,
        "irodori": irodori,
    }


def play_worker(chunks: "queue.Queue[bytes | None]", *, no_play: bool) -> None:
    while True:
        audio = chunks.get()
        if audio is None:
            return
        if no_play:
            continue
        subprocess.run(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "error", "-i", "-"],
            input=audio,
            check=False,
        )


def iter_sse(response: Any):
    event = "message"
    data_lines: list[str] = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())


def main() -> int:
    args = parse_args()
    payload = json.dumps(request_payload(args), ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"

    request = urllib.request.Request(args.url, data=payload, headers=headers, method="POST")
    chunks: queue.Queue[bytes | None] = queue.Queue()
    player = threading.Thread(target=play_worker, args=(chunks,), kwargs={"no_play": args.no_play})
    player.start()

    count = 0
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            print(f"connected: {response.status} {response.headers.get('content-type')}", file=sys.stderr)
            for event, data in iter_sse(response):
                body = json.loads(data)
                if event == "speech.audio.delta":
                    count += 1
                    audio = base64.b64decode(body["audio"])
                    if args.save_prefix:
                        path = f"{args.save_prefix}-{count:03d}.{args.response_format}"
                        with open(path, "wb") as f:
                            f.write(audio)
                        print(f"chunk {count}: {len(audio)} bytes -> {path}", file=sys.stderr)
                    else:
                        print(f"chunk {count}: {len(audio)} bytes", file=sys.stderr)
                    chunks.put(audio)
                elif event == "speech.audio.done":
                    print("done", file=sys.stderr)
                    break
                elif event == "error":
                    print(data, file=sys.stderr)
                    return 1
                else:
                    print(f"ignored event {event}: {data}", file=sys.stderr)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    finally:
        chunks.put(None)
        player.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

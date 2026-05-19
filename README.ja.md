# Irodori-TTS-MLX-Server 日本語 README

[English README](README.md)

[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX) を OpenAI 互換の
`/v1/audio/speech` API から使うための、Apple Silicon / MLX 向けローカル TTS
サーバーです。PyTorch / CUDA 版の upstream server の全機能互換ではなく、
Irodori-TTS-MLX で実装・検証済みの機能に絞った Apple Silicon 向けサーバーです。

現在の公開 API は次の通りです。

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- Irodori 固有の chunk-level Server-Sent Events 用
  `POST /v1/audio/speech/stream-chunks`
- `/v1/audio/voices` 配下の `GET`, `POST`, `GET by id`, `PUT`, `DELETE`

モデル重みを設定しなくてもパッケージは import できます。その場合、`/health` は
`speech_runtime.load_state` を返し、音声生成は `runtime_unavailable` の OpenAI 形式
エラーを返します。実生成には `IRODORI_MLX_WEIGHTS_REPO` または
`IRODORI_MLX_WEIGHTS_DIR` で変換済み Irodori-TTS-MLX weights layout を指定します。

## ドキュメント

- [docs/real_model_setup.md](docs/real_model_setup.md): 初回セットアップ、変換済み
  weights layout、ローカル起動、speech smoke、よくあるエラー。
- [docs/openai_client_examples.md](docs/openai_client_examples.md): `curl` と Python
  OpenAI client の例、bearer auth、FFmpeg 形式、OpenAI speech streaming 非対応の扱い、
  Irodori 固有の chunk-level SSE extension。
- [docs/deployment.md](docs/deployment.md): Apple Silicon ローカル運用、launchd、
  bearer auth、queue、health check、ログ、環境変数。
- [docs/upstream_compatibility.md](docs/upstream_compatibility.md): upstream server
  との差分と対応状況。
- [docs/mvp_scope.md](docs/mvp_scope.md): このリポジトリで公開する MVP 境界と non-goals。

## ローカル開発

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

サーバーを起動します。

```bash
python -m irodori_tts_mlx_server --host 127.0.0.1 --port 8000
```

console script でも同じです。

```bash
irodori-tts-mlx-server --host 127.0.0.1 --port 8000
```

health check:

```bash
curl http://127.0.0.1:8000/health
```

## 実モデルでの起動

Irodori-TTS-MLX runtime を同じ仮想環境へ入れてから起動します。

```bash
python -m pip install -e /path/to/Irodori-TTS-MLX"[runtime]"
```

hosted converted weights を使う例です。

```bash
python -m irodori_tts_mlx_server \
  --weights-repo t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
  --host 127.0.0.1 \
  --port 8000
```

ローカルの hosted layout コピーを使う場合は `--weights-dir` を指定します。

```bash
python -m irodori_tts_mlx_server \
  --weights-dir /opt/irodori-models/voicedesign-v2 \
  --host 127.0.0.1 \
  --port 8000
```

同じ設定は環境変数 `IRODORI_MLX_WEIGHTS_REPO`, `IRODORI_MLX_WEIGHTS_DIR`,
`IRODORI_MLX_WEIGHTS_REVISION` でも指定できます。`IRODORI_MLX_PRELOAD=1` を
設定すると起動時にモデルを load します。未設定の場合は最初の speech request で lazy
load します。

変換済み layout には `irodori_mlx_manifest.json`, `model_config.json`,
`tokenizer_config.json`, `conversion_metadata.json`, `weights.npz`,
`checksums.sha256` が必要です。

## API 例

model 一覧:

```bash
curl http://127.0.0.1:8000/v1/models
```

VoiceDesign v2 no-reference / caption で WAV を生成します。

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"model":"irodori-tts-mlx","input":"こんにちは。","voice":"voicedesign","response_format":"wav","irodori":{"no_reference":true,"caption":"落ち着いた明瞭なナレーション","preset":"balanced"}}' \
  --output speech.wav
```

`wav` と `pcm` は追加エンコーダなしで使えます。`mp3`, `flac`, `opus`, `aac` は
サーバーホストに FFmpeg が必要です。OpenAI 互換の `/v1/audio/speech` route は
synthesis streaming を行いません。`stream=true`, `stream_format`,
`Accept: text/event-stream` を送ると `unsupported_streaming` エラーを返します。

よく使う `irodori` option は `no_reference`, `caption`, `preset`, `seconds`,
`duration_scale`, `num_steps`, `seed`, `chunking`, `chunk_max_chars`,
`tail_trim_ms`, `tail_silence_trim_ms` です。OpenAI の `speed` は
`duration_scale` 未指定時に `duration_scale=1/speed` として扱います。
`preset` は `ultra-fast`, `fast`, `balanced`, `quality` を受け付け、それぞれ
8, 12, 24, 40 sampling steps に対応します。`ultra-fast` は、`seconds` と
`duration_scale` が未指定かつ `speed=1.0` の場合、対応する Irodori-TTS-MLX
runtime に短文向け auto-duration cap も渡します。
管理対象 reference voice の短文リクエストで `fast` または `ultra-fast` を使い、
`seconds` と `duration_scale` を省略し `speed=1.0` のままにした場合、サーバーは
保守的な文字数ベースの `seconds` 推定値を自動設定します。低遅延応答で明らかな
過生成を避けるための挙動で、長文や明示的な duration 指定は末尾切れを避けるため
変更しません。

低遅延に再生を始めたい用途向けに、Irodori 固有 extension として
`POST /v1/audio/speech/stream-chunks` も提供します。同じ speech request 形式と
chunking 設定を使い、生成済みのテキスト chunk ごとに `audio_chunk` event、最後に
`done` event を Server-Sent Events で返します。

```bash
curl -N http://127.0.0.1:8000/v1/audio/speech/stream-chunks \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'Authorization: Bearer <token>' \
  -d '{"model":"irodori-tts-mlx","input":"最初の文です。次の文です。","voice":"voicedesign","response_format":"wav","irodori":{"no_reference":true,"caption":"落ち着いた明瞭なナレーション","chunking":true,"chunk_max_chars":80}}'
```

```text
event: audio_chunk
data: {"index":0,"text":"最初の文です。","format":"wav","media_type":"audio/wav","audio_base64":"..."}

event: done
data: {"chunks":1}
```

`audio_base64` には chunk 単位の完全な音声ファイルが入ります。client は各
`audio_chunk` を decode して再生 queue に積みながら、後続 chunk の生成を待てます。
この endpoint は OpenAI speech API の互換 contract ではなく、この server の追加機能です。

## 管理対象 reference voice

`IRODORI_SERVER_VOICES_DIR` 配下に reference voice を保存できます。対応拡張子は
`.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.aac`, `.webm` です。
speech request の `voice` に管理対象 ID または `{"id":"<voice_id>"}` を指定すると、
サーバーが `irodori.reference_wav` と `irodori.no_reference=false` を補います。

任意のローカルパス、path traversal、remote URL、symlink、latent `.pt` / `.pth` は
受け付けません。upload 上限は `IRODORI_SERVER_MAX_VOICE_UPLOAD_BYTES` で、既定は 50 MiB
です。

```bash
curl http://127.0.0.1:8000/v1/audio/voices \
  -H 'Authorization: Bearer <token>'

curl http://127.0.0.1:8000/v1/audio/voices \
  -H 'Authorization: Bearer <token>' \
  -F voice_id=sample \
  -F file=@sample.wav
```

## 運用設定

既定では認証なしでローカル開発できます。`IRODORI_SERVER_BEARER_TOKEN` を設定すると
`/v1/*` に `Authorization: Bearer <token>` が必要になります。`/health` は local probe
用に認証不要です。`IRODORI_API_KEY` は互換 alias として使えます。

MLX synthesis は重いため、既定では同時実行数を 1 に制限します。
`IRODORI_SERVER_MAX_CONCURRENT_SYNTHESIS=1` と
`IRODORI_SERVER_QUEUE_TIMEOUT_SECONDS=30` が既定値です。timeout した request は
`synthesis_queue_timeout` を返します。

## 検証

```bash
pytest
ruff check .
```

実モデル smoke は opt-in です。

```bash
IRODORI_REAL_MLX_SMOKE=1 \
IRODORI_MLX_WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign \
pytest -m real_mlx tests/test_real_mlx_smoke.py
```

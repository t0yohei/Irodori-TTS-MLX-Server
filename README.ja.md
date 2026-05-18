# Irodori-TTS-MLX-Server 日本語 README

[English README](README.md)

[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX) を OpenAI 互換の
`/v1/audio/speech` API から使うための、Apple Silicon / MLX 向けローカル TTS
サーバーです。PyTorch / CUDA 版の upstream server をそのまま置き換えるものではなく、
MLX ランタイムで安全に扱える範囲を明示して実装しています。

現在の公開 API は次の通りです。

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `/v1/audio/voices` 配下の `GET`, `POST`, `GET by id`, `PUT`, `DELETE`

モデル重みを設定しなくてもパッケージは import できます。その場合、`/health` は
`speech_runtime.load_state` を返し、音声生成は `runtime_unavailable` の OpenAI 形式
エラーを返します。実生成には `IRODORI_MLX_WEIGHTS_REPO` または
`IRODORI_MLX_WEIGHTS_DIR` で変換済み Irodori-TTS-MLX weights layout を指定します。

## ドキュメント

- [docs/real_model_setup.md](docs/real_model_setup.md): 初回セットアップ、変換済み
  weights layout、ローカル起動、speech smoke、よくあるエラー。
- [docs/openai_client_examples.md](docs/openai_client_examples.md): `curl` と Python
  OpenAI client の例、bearer auth、FFmpeg 形式、streaming 非対応の扱い。
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
サーバーホストに FFmpeg が必要です。streaming synthesis / SSE は未対応で、
`unsupported_streaming` エラーを返します。

よく使う `irodori` option は `no_reference`, `caption`, `preset`, `seconds`,
`duration_scale`, `num_steps`, `seed`, `chunking`, `chunk_max_chars`,
`tail_trim_ms`, `tail_silence_trim_ms` です。OpenAI の `speed` は
`duration_scale` 未指定時に `duration_scale=1/speed` として扱います。

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

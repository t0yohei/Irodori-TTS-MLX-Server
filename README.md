# Irodori-TTS-MLX-Server

OpenAI-compatible local TTS server for
[Irodori-TTS-MLX](https://github.com/t0yohei/Irodori-TTS-MLX).

The initial implementation is intentionally scoped to a small MVP: expose
`POST /v1/audio/speech` for VoiceDesign v2 no-reference/caption generation
through an MLX-backed runtime. See [docs/mvp_scope.md](docs/mvp_scope.md) for
the current API target, non-goals, and follow-up implementation boundaries.

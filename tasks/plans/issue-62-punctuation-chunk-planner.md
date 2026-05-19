# Issue 62 implementation plan

## Goal
- Add punctuation-oriented automatic chunk planning for speech generation without breaking existing `chunk_max_chars` behavior.

## Completion criteria
- `chunk_mode="punctuation"` can chunk normal Japanese/mixed text without manual `chunk_max_chars` tuning.
- The tested sample keeps `これは stream chunks の動作確認です。` as one chunk.
- Short punctuation segments can merge by `chunk_min_chars` / `chunk_target_chars`.
- Long punctuation-free segments still hard split by `chunk_hard_max_chars`.
- Existing `chunk_max_chars` behavior and tests continue to pass.

## Verification
- `uv run --extra dev pytest tests/test_runtime_adapter.py -k chunk`
- `uv run --extra dev pytest tests/test_openai_api.py -k stream_chunks`
- `uv run --extra dev ruff check src tests`
- `uv run --extra dev pytest` if focused tests pass.

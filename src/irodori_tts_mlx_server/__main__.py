"""Command-line entry point for local development."""

from __future__ import annotations

import argparse
import os

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Irodori-TTS-MLX FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    parser.add_argument("--weights", help="Path to converted Irodori-TTS-MLX .npz weights.")
    parser.add_argument("--model-config-json", help="Path to the matching model_config.json.")
    parser.add_argument("--weights-dir", help="Local hosted converted weights layout directory.")
    parser.add_argument("--weights-repo", help="Hosted converted weights repository id.")
    parser.add_argument("--weights-revision", help="Hosted converted weights revision.")
    parser.add_argument("--preload", action="store_true", help="Load the MLX runtime during startup.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.weights:
        os.environ["IRODORI_MLX_WEIGHTS_PATH"] = args.weights
    if args.model_config_json:
        os.environ["IRODORI_MLX_MODEL_CONFIG_JSON"] = args.model_config_json
    if args.weights_dir:
        os.environ["IRODORI_MLX_WEIGHTS_DIR"] = args.weights_dir
    if args.weights_repo:
        os.environ["IRODORI_MLX_WEIGHTS_REPO"] = args.weights_repo
    if args.weights_revision:
        os.environ["IRODORI_MLX_WEIGHTS_REVISION"] = args.weights_revision
    if args.preload:
        os.environ["IRODORI_MLX_PRELOAD"] = "1"
    uvicorn.run(
        "irodori_tts_mlx_server.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()

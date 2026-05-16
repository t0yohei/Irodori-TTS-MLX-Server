"""Audio response format conversion helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from irodori_tts_mlx_server.runtime import SpeechGenerationResult


SUPPORTED_RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

FFMPEG_FORMATS = {
    "mp3": "mp3",
    "opus": "opus",
    "aac": "adts",
    "flac": "flac",
}


@dataclass(frozen=True)
class AudioConversionError(RuntimeError):
    message: str
    code: str

    def __str__(self) -> str:
        return self.message


def convert_audio_response(
    result: SpeechGenerationResult, response_format: str
) -> SpeechGenerationResult:
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise AudioConversionError(
            f"response_format '{response_format}' is not supported. Supported formats: "
            f"{', '.join(SUPPORTED_RESPONSE_FORMATS)}.",
            code="unsupported_response_format",
        )
    if response_format == "wav":
        return SpeechGenerationResult(audio=result.audio, media_type=MEDIA_TYPES["wav"])
    if response_format == "pcm":
        return SpeechGenerationResult(audio=_wav_to_pcm(result.audio), media_type=MEDIA_TYPES["pcm"])
    return SpeechGenerationResult(
        audio=_convert_with_ffmpeg(result.audio, response_format),
        media_type=MEDIA_TYPES[response_format],
    )


def _wav_to_pcm(wav_audio: bytes) -> bytes:
    try:
        with wave.open(_BytesReader(wav_audio), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise AudioConversionError(
                    "PCM response_format requires an uncompressed WAV runtime output.",
                    code="unsupported_response_format",
                )
            return wav_file.readframes(wav_file.getnframes())
    except AudioConversionError:
        raise
    except (EOFError, wave.Error) as exc:
        raise AudioConversionError(
            "Runtime output could not be converted to response_format='pcm' because it is not "
            "a valid WAV file.",
            code="audio_conversion_failed",
        ) from exc


def _convert_with_ffmpeg(wav_audio: bytes, response_format: str) -> bytes:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise AudioConversionError(
            f"response_format='{response_format}' requires FFmpeg. Install FFmpeg or request "
            "response_format='wav' or 'pcm'.",
            code="response_format_unavailable",
        )

    suffix = f".{response_format}"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.wav"
        output_path = Path(tmpdir) / f"output{suffix}"
        input_path.write_bytes(wav_audio)
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-f",
            FFMPEG_FORMATS[response_format],
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "FFmpeg exited without an error message."
            raise AudioConversionError(
                f"FFmpeg could not encode response_format='{response_format}': {detail}",
                code="audio_conversion_failed",
            )
        try:
            return output_path.read_bytes()
        except OSError as exc:
            raise AudioConversionError(
                f"FFmpeg did not create response_format='{response_format}' output.",
                code="audio_conversion_failed",
            ) from exc


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._position
        chunk = self._data[self._position : self._position + size]
        self._position += len(chunk)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = len(self._data) + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self._position = max(0, position)
        return self._position

    def tell(self) -> int:
        return self._position

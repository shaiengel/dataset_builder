import io
from pathlib import Path

from pydub import AudioSegment


def convert_mp3_to_wav(
    mp3_bytes: bytes,
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """Convert MP3 bytes to WAV file (mono, 16kHz)."""
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    audio.export(output_path, format="wav")
    return output_path


def truncate_wav(
    wav_path: Path,
    truncate_at_seconds: float,
) -> None:
    """Truncate WAV file at the specified timestamp (in-place)."""
    audio = AudioSegment.from_wav(wav_path)
    truncate_at_ms = int(truncate_at_seconds * 1000)
    truncated = audio[:truncate_at_ms]
    truncated.export(wav_path, format="wav")

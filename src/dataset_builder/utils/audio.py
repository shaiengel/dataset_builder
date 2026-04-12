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

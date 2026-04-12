from dataclasses import dataclass
from enum import Enum

from stable_whisper.result import Segment


class AlignmentStatus(Enum):
    OK = "ok"
    TRUNCATED = "truncated"
    NO_DATA = "no_data"


@dataclass
class SegmentResult:
    """Result of segment parsing with alignment status."""
    segments: list[Segment]
    status: AlignmentStatus
    truncate_at: float | None  # Timestamp where audio should be truncated (if status is TRUNCATED)
    total_words_aligned: int
    mismatch_info: str | None  # Details about the mismatch if truncated

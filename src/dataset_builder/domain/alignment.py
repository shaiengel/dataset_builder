from dataclasses import dataclass


@dataclass
class AlignedWord:
    """A word matched between VTT and JSON transcript."""
    word: str
    start: float
    end: float
    probability: float
    vtt_cue_index: int


@dataclass
class AlignmentResult:
    """Result of aligning VTT with JSON transcript."""
    words: list[AlignedWord]
    truncated: bool
    truncation_reason: str | None

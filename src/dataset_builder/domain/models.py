from dataclasses import dataclass


@dataclass
class Word:
    """A single word from stable_whisper JSON."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Transcript:
    """Flattened list of all words from JSON."""
    words: list[Word]


@dataclass
class VttCue:
    """A single VTT cue with timing and text."""
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Vtt:
    """Parsed VTT file."""
    cues: list[VttCue]

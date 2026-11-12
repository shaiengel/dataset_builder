import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inject_space_to_dataset import restore_timestamp_spaces  # noqa: E402

from dataset_builder.domain.models import Transcript, Vtt, VttCue, Word
from dataset_builder.infrastructure.segment_parser import SegmentParser
from dataset_builder.infrastructure.whisper_dataset_generator import WhisperDatasetGenerator


def _word(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end, probability=1.0)


def test_segment_text_has_leading_space_and_no_trailing_space():
    transcript = Transcript(words=[_word(" שלום", 0.0, 0.5), _word(" עולם", 0.5, 1.0)])
    vtt = Vtt(cues=[VttCue(start=0.0, end=1.0, text="שלום עולם")])

    result = SegmentParser().parse(transcript, vtt)

    assert [seg.text for seg in result.segments] == [" שלום עולם"]


def test_slice_transcript_matches_whisper_layout():
    gen = WhisperDatasetGenerator()
    slices = [{"seek": 0.0, "segments": [{"start": 0.0, "end": 1.0, "text": " שלום עולם", "word_scores": [1.0]}]}]

    class _Loader:
        def next_chunk(self, *_):
            raise AssertionError("audio not needed")

    gen._get_slice_audio_data = lambda *_: b""
    example = next(gen._generate_examples_from_slices(slices, 30, _Loader(), {"source_id": "x"}))

    assert example["transcript"] == "<|0.00|> שלום עולם<|1.00|>"


def test_restore_fixes_old_trailing_space_layout():
    old = ["<|0.52|>ילד שנמצא באופקים, <|3.42|><|3.98|>ההורה יקבל <|5.94|>"]

    assert restore_timestamp_spaces(old) == {
        "transcript": ["<|0.52|> ילד שנמצא באופקים,<|3.42|><|3.98|> ההורה יקבל<|5.94|>"]
    }


def test_restore_is_idempotent_on_correct_layout():
    good = ["<|0.00|> already spaced<|1.00|><|1.02|> still text<|2.00|>"]

    assert restore_timestamp_spaces(good) == {"transcript": good}

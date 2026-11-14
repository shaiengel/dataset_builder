import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inject_space_to_dataset import restore_timestamp_spaces  # noqa: E402

from dataset_builder.domain.models import Transcript, Vtt, VttCue, Word
from dataset_builder.infrastructure.segment_parser import SegmentParser
from dataset_builder.infrastructure.whisper_dataset_generator import WhisperDatasetGenerator, normalize_asr_text


def _word(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end, probability=1.0)


def test_segment_text_has_leading_space():
    transcript = Transcript(words=[_word(" hello", 0.0, 0.5), _word(" world", 0.5, 1.0)])
    vtt = Vtt(cues=[VttCue(start=0.0, end=1.0, text="hello world")])

    result = SegmentParser().parse(transcript, vtt)

    assert [seg.text for seg in result.segments] == [" hello world"]


def test_slice_transcript_is_normalized_with_leading_space():
    gen = WhisperDatasetGenerator()
    slices = [{"seek": 0.0, "segments": [{"start": 0.0, "end": 1.0, "text": " hello world", "word_scores": [1.0]}]}]

    class _Loader:
        def next_chunk(self, *_):
            raise AssertionError("audio not needed")

    gen._get_slice_audio_data = lambda *_: b""
    example = next(gen._generate_examples_from_slices(slices, 30, _Loader(), {"source_id": "x"}))

    assert example["transcript"] == "<|0.00|> hello world<|1.00|>"


def test_restore_fixes_old_trailing_space_layout():
    old = ["<|0.52|>good morning, <|3.42|><|3.98|>how are you <|5.94|>"]

    assert restore_timestamp_spaces(old) == {
        "transcript": ["<|0.52|> good morning,<|3.42|><|3.98|> how are you<|5.94|>"]
    }


def test_restore_is_idempotent_on_correct_layout():
    good = ["<|0.00|> already spaced<|1.00|><|1.02|> still text<|2.00|>"]

    assert restore_timestamp_spaces(good) == {"transcript": good}


def test_normalize_asr_text_does_not_strip_boundary_spaces():
    assert normalize_asr_text(" hello  world ") == " hello world "


def test_normalize_asr_text_normalizes_punctuation():
    assert normalize_asr_text("hello — world…") == "hello - world..."


def test_normalize_asr_text_drops_non_whitelisted_chars():
    assert normalize_asr_text("abc 123 test") == "abc 123 test"

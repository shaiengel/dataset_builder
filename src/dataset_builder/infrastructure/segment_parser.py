import logging
from dataclasses import dataclass

from stable_whisper.result import Segment, WordTiming

from dataset_builder.domain.models import Transcript, Vtt, Word
from dataset_builder.domain.segment_result import AlignmentStatus, SegmentResult

logger = logging.getLogger(__name__)


@dataclass
class AlignmentState:
    segments: list[list[WordTiming]]
    total_aligned: int
    status: AlignmentStatus
    truncate_at: float | None
    mismatch_info: str | None


class SegmentParser:
    """Parses JSON transcript and VTT into stable_whisper Segments with alignment."""

    def parse(self, transcript: Transcript, vtt: Vtt) -> SegmentResult:
        """Parse and align transcript with VTT, returning Segments."""
        validation_result = self._validate_input(transcript, vtt)
        if validation_result:
            return validation_result

        vtt_words = self._extract_vtt_words(vtt)
        num_cues = len(vtt.cues)
        state = self._align_words(transcript.words, vtt_words, num_cues)
        state = self._check_length_mismatch(state, transcript.words, vtt_words)
        segments = self._convert_list_to_segments(state.segments)

        return SegmentResult(
            segments=segments,
            status=state.status,
            truncate_at=state.truncate_at,
            total_words_aligned=state.total_aligned,
            mismatch_info=state.mismatch_info,
        )

    def _validate_input(self, transcript: Transcript, vtt: Vtt) -> SegmentResult | None:
        """Validate inputs, return error result if invalid."""
        if not transcript or not transcript.words:
            return SegmentResult(
                segments=[],
                status=AlignmentStatus.NO_DATA,
                truncate_at=None,
                total_words_aligned=0,
                mismatch_info="No transcript data",
            )

        if not vtt or not vtt.cues:
            return SegmentResult(
                segments=[],
                status=AlignmentStatus.NO_DATA,
                truncate_at=None,
                total_words_aligned=0,
                mismatch_info="No VTT data",
            )

        return None

    def _extract_vtt_words(self, vtt: Vtt) -> list[tuple[str, int]]:
        """Extract words from VTT cues with their cue index."""
        vtt_words = []
        for cue_idx, cue in enumerate(vtt.cues):
            for word in cue.text.split():
                vtt_words.append((word.strip(), cue_idx))
        return vtt_words

    def _align_words(
        self, json_words: list[Word], vtt_words: list[tuple[str, int]], num_cues: int
    ) -> AlignmentState:
        """Align JSON words with VTT words, stop on mismatch."""
        segments: list[list[WordTiming]] = [[] for _ in range(num_cues)]
        total_aligned = 0
        min_len = min(len(json_words), len(vtt_words))

        for i in range(min_len):
            json_word = json_words[i]
            vtt_word, cue_idx = vtt_words[i]

            json_text = json_word.word.strip()
            vtt_text = vtt_word.strip()

            if self._normalize(json_text) != self._normalize(vtt_text):
                logger.warning(
                    f"Mismatch at word {i}: JSON='{json_text}' vs VTT='{vtt_text}'. "
                    f"Truncating at timestamp {json_word.start:.3f}s"
                )
                return AlignmentState(
                    segments=segments,
                    total_aligned=total_aligned,
                    status=AlignmentStatus.TRUNCATED,
                    truncate_at=json_word.start,
                    mismatch_info=f"Word {i}: '{json_text}' vs '{vtt_text}'",
                )

            word_timing = self._create_word_timing(json_word)
            segments[cue_idx].append(word_timing)
            total_aligned += 1

        return AlignmentState(
            segments=segments,
            total_aligned=total_aligned,
            status=AlignmentStatus.OK,
            truncate_at=None,
            mismatch_info=None,
        )

    def _check_length_mismatch(
        self,
        state: AlignmentState,
        json_words: list[Word],
        vtt_words: list[tuple[str, int]],
    ) -> AlignmentState:
        """Check if VTT is shorter than transcript and update state."""
        if state.status != AlignmentStatus.OK:
            return state

        if len(json_words) > len(vtt_words):
            logger.warning(
                f"VTT shorter than transcript ({len(vtt_words)} vs {len(json_words)} words). "
                f"Truncating at word {len(vtt_words)}"
            )
            return AlignmentState(
                segments=state.segments,
                total_aligned=state.total_aligned,
                status=AlignmentStatus.TRUNCATED,
                truncate_at=json_words[len(vtt_words)].start,
                mismatch_info=f"VTT shorter: {len(vtt_words)} vs {len(json_words)} words",
            )

        return state

    def _create_word_timing(self, word: Word) -> WordTiming:
        """Create a WordTiming from a Word."""
        return WordTiming(
            word=f"{word.word.strip()} ",
            start=word.start,
            end=word.end,
            probability=word.probability,
        )

    def _convert_list_to_segments(self, word_timings_list: list[list[WordTiming]]) -> list[Segment]:
        """Build Segment objects from list of word timings."""
        segments = []
        for word_timings in word_timings_list:
            if word_timings:
                segments.append(Segment(words=word_timings))
        return segments

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

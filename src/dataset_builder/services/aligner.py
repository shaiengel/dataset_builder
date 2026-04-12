import logging

from dataset_builder.domain.alignment import AlignedWord, AlignmentResult
from dataset_builder.domain.models import Transcript, Vtt

logger = logging.getLogger(__name__)


class Aligner:
    def align(self, transcript: Transcript, vtt: Vtt) -> AlignmentResult:
        """Align words from JSON transcript with VTT cues.

        If VTT is shorter, truncate transcript to match.
        If mismatch found, log warning and truncate both at that point.
        """
        # Extract words from VTT cues
        vtt_words = []
        for cue_idx, cue in enumerate(vtt.cues):
            for word in cue.text.split():
                vtt_words.append((word.strip(), cue_idx))

        aligned = []
        truncated = False
        truncation_reason = None

        json_words = transcript.words
        min_len = min(len(json_words), len(vtt_words))

        for i in range(min_len):
            json_word = json_words[i]
            vtt_word, cue_idx = vtt_words[i]

            # Normalize for comparison (strip whitespace, compare lowercase)
            json_text = json_word.word.strip()
            vtt_text = vtt_word.strip()

            if self._normalize(json_text) != self._normalize(vtt_text):
                logger.warning(
                    f"Mismatch at index {i}: JSON='{json_text}' vs VTT='{vtt_text}'. "
                    f"Truncating alignment at this point."
                )
                truncated = True
                truncation_reason = f"mismatch at index {i}: '{json_text}' vs '{vtt_text}'"
                break

            aligned.append(
                AlignedWord(
                    word=json_text,
                    start=json_word.start,
                    end=json_word.end,
                    probability=json_word.probability,
                    vtt_cue_index=cue_idx,
                )
            )

        if not truncated and len(json_words) > len(vtt_words):
            truncated = True
            truncation_reason = f"VTT shorter than transcript ({len(vtt_words)} vs {len(json_words)} words)"
            logger.warning(
                f"VTT has fewer words than transcript. "
                f"Truncating transcript from {len(json_words)} to {len(vtt_words)} words."
            )

        return AlignmentResult(
            words=aligned,
            truncated=truncated,
            truncation_reason=truncation_reason,
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

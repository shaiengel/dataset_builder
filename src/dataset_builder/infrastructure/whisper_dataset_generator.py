import logging
from pathlib import Path
import uuid
from typing import Iterator

from datasets import (
    Audio as AudioColumnType,
    Dataset,
    Features,
    Value as ValueColumnType,
)
from stable_whisper.result import Segment
from stable_whisper.audio import AudioLoader
from audiosample import AudioSample

from dataset_builder.domain.dataset_generator import DatasetGenerator
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

WHISPER_EXPECTED_SAMPLE_RATE = 16000


class WhisperDatasetGenerator(DatasetGenerator):
    def __init__(self) -> None:
        self._sample_rate = WHISPER_EXPECTED_SAMPLE_RATE    
    
    def _get_segment_word_scores(self, segment: Segment) -> list[float]:
        """
        Get the word scores for a segment.
        This is a helper function to extract the word scores from a segment.
        """
        if not segment.has_words:
            return []

        # Extract word scores from the segment
        word_scores = []
        for word in segment.words:
            if hasattr(word, "probability"):
                word_scores.append(word.probability)
        return word_scores
    
    def _calculate_median_quality_score(self, scores: list[float]) -> float:
        """
        Calculate the median quality score for a list of scores.
        This is a helper function to calculate the median quality score for a list of scores.
        """
        # Calculate the median probability of all words in the segment
        quality_score = float(np.median(scores)) if scores else 0.0
        return quality_score


    def _calculate_segments_quality_score(self, segments: list[Segment]) -> float:
        if not segments:
            return 0.0

        """Calculate the quality score based on the median word probabilities for a single segment."""
        try:
            all_word_probs = []
            for segment in segments:
                all_word_probs.extend(self._get_segment_word_scores(segment))
            # Calculate the median probability of all words in the segment
            quality_score = self._calculate_median_quality_score(all_word_probs)
            return quality_score

        except Exception:
            return 0.0
    
    def _generate_slices(
        self,
        input_segments: list[Segment],
        audio_duration: float,
        slice_length: int,
        per_segment_quality_threshold: float = 0,
    ):
        next_slice_start = 0
        curr_input_segment_idx = 0
        slices = []
        while next_slice_start < audio_duration:
            slice_start = next_slice_start

            # Ensure current segment exists
            # and validate it's duration.
            if curr_input_segment_idx < len(input_segments):
                curr_input_segment_duration = (
                    input_segments[curr_input_segment_idx].end - input_segments[curr_input_segment_idx].start
                )
                # If the first segment to work on is too long for a single slice or of 0 length we must skip it.
                if curr_input_segment_duration > slice_length or curr_input_segment_duration == 0:
                    # skip if any segment ahead
                    if curr_input_segment_idx + 1 < len(input_segments):
                        next_slice_start = input_segments[curr_input_segment_idx + 1].start
                        curr_input_segment_idx += 1
                    # or break since nothing more to work on
                    else:
                        next_slice_start = audio_duration

                    continue

            curr_slice_source_segment_idxs = []
            curr_slice_source_segments = []
            curr_slice_segments = []
            curr_slice = {"segments": curr_slice_segments, "seek": slice_start}
            slices.append(curr_slice)
            # normal slice length is the expected slice hop - but this could be overridden below. See comments.
            next_slice_start = slice_start + slice_length
            # clip the slice end to the total audio duration
            slice_end = min(next_slice_start, audio_duration)

            # While more segments to work on and the current segment start is within the slice
            while curr_input_segment_idx < len(input_segments) and input_segments[curr_input_segment_idx].start < slice_end:
                curr_input_segment = input_segments[curr_input_segment_idx]

                # track the source segments used in this slice for quality analysis after slice completion
                curr_slice_source_segments.append(curr_input_segment)
                curr_slice_source_segment_idxs.append(curr_input_segment_idx)

                # Add it to the slice
                slice_segment = {
                    "start": max(0, curr_input_segment.start - slice_start),  # relative to slice
                }
                curr_slice_segments.append(slice_segment)

                # Clip the segment end to the entire audio duration
                # This is meant to prevent small segment timing overflows over audio
                # duration which stems from arbitrary rounding errors in the data prep
                # and subtitles alignment logic.
                curr_input_segment_end = min(curr_input_segment.end, audio_duration)

                # If this input segment ends within the slice
                # It would be entirely contained including it's text and timestamps
                if curr_input_segment_end <= slice_end:
                    #   s   e   s         e
                    #  /    \  /          \??????
                    # |_________________________|
                    #                     ^
                    slice_segment["end"] = min(slice_length, curr_input_segment_end - slice_start)  # relative to slice
                    slice_segment["text"] = curr_input_segment.text
                    slice_segment["word_scores"] = self._get_segment_word_scores(curr_input_segment)

                    # entire segment is included - no need to reference it again on the next slice.
                    curr_input_segment_idx += 1

                # Else - we cannot complete this segment on this slice.
                # The "start" of the segment is kept in the slice to mark it's crossing onto the next
                # slice but the next slice will also need to start at the **end** of the previous segment
                # to allow proper "restart" of the overflowing segment
                else:
                    # This slice ends - close this slice

                    # remove the last added source segment - it was not used
                    curr_slice_source_segments = curr_slice_source_segments[:-1]
                    curr_slice_source_segment_idxs = curr_slice_source_segment_idxs[:-1]

                    # Special case - If the "start only" segment is the only one - don't include it at all.
                    # Instead, this slice would be left empty.
                    if len(curr_slice_segments) == 1:
                        #           s                    e
                        #          /                     \
                        # |_________________________||........
                        #                                ^
                        curr_slice_segments.clear()
                        
                        # In this special case, the current segment starts within
                        # the slice and ends outside of it. But it is the only segment.
                        # We need to start the next slice on the **start** of this segment
                        # and not at the end of the previous one (which is not within this slice
                        # at all
                        next_slice_start = input_segments[curr_input_segment_idx].start
                    else:
                        #   s    e  s                    e
                        #  /     \ /                     \
                        # |_________________________||........
                        #                                ^
                        # This is the normal cross-over case.
                        # The current segment starts within this slice
                        # and ends outside of it and other segments within this slice were closed normally.
                        # We need to start the next slice on the **end** of prev segment before the "start-only" one.
                        next_slice_start = input_segments[curr_input_segment_idx - 1].end


                    # Break, this slice is done.
                    break

            # Slice Quality Control
            slice_quality_score = self._calculate_segments_quality_score(curr_slice_source_segments)

            # Check if the slice quality is below threshold to abandon it and force a new slice
            if curr_slice_source_segments and slice_quality_score < per_segment_quality_threshold:
                # This slice is suspected as low quality

                # Look for a segment with good quality to start the next slice
                # skip the first segment in the slice (otherwise we probably are going
                # to just repeat the same slice)
                found_good_segment = False
                for seg_idx_within_slice, seg_of_slice in enumerate(curr_slice_source_segments):
                    if seg_idx_within_slice == 0:
                        continue

                    segment_score = self._calculate_segments_quality_score([seg_of_slice])

                    if segment_score >= per_segment_quality_threshold:
                        # Found a good enough segment, start next slice from here
                        next_slice_start = seg_of_slice.start
                        curr_input_segment_idx = curr_slice_source_segment_idxs[seg_idx_within_slice]
                        found_good_segment = True
                        break

                # If no good segment found, start from the end of the last checked segment
                if not found_good_segment:
                    next_segment_idx_after_slice_segments = curr_slice_source_segment_idxs[-1] + 1
                    # if any segment ahead
                    if next_segment_idx_after_slice_segments < len(input_segments):
                        next_slice_start = input_segments[next_segment_idx_after_slice_segments].start
                        curr_input_segment_idx = next_segment_idx_after_slice_segments
                    # or there are more segments - stop slicing
                    else:
                        next_slice_start = audio_duration

                # Clear the current slice content as we're abandoning it
                curr_slice_segments.clear()

        return slices
    
    def _merge_slice_segments(self, slices: list[dict], merge_below_gap_threshold: float = 0.3) -> list[dict]:
        """
        Merge segments within each slice that are close together.

        Args:
            slices: List of slices, each containing a list of segments
            merge_below_gap_threshold: Merge segments if gap between them is less than this threshold (in seconds)

        Returns:
            List of slices with merged segments
        """
        if not slices:
            return slices

        result_slices = []

        for slice_data in slices:
            # Create a new slice with the same properties as the original, but copy it to avoid modifying the original
            new_slice = {key: value for key, value in slice_data.items() if key != "segments"}
            new_slice["segments"] = []

            segments = slice_data.get("segments", [])

            # If no segments or only one segment, no merging needed
            if len(segments) <= 1:
                new_slice["segments"] = [segment.copy() for segment in segments]
                result_slices.append(new_slice)
                continue

            # Create a copy of segments to process
            result_segments = [segment.copy() for segment in segments]

            # Process segments in reverse order
            i = len(result_segments) - 1
            while i > 0:  # Stop at index 1 (second segment)
                current_segment = result_segments[i]
                prev_segment = result_segments[i - 1]

                # Check if we can merge the current segment with the previous one
                can_merge = False

                # Current segment must have start, end, and text to be mergeable
                # Note: No "end" cases means an open-only slice where the last segment
                # mark a segment which could not end within the same slice. we need
                # to keep it as is.
                if all(key in current_segment for key in ["start", "end", "text"]):

                    # Calculate the gap between segments
                    gap = current_segment["start"] - prev_segment["end"]

                    # Check if the gap is small enough
                    if gap < merge_below_gap_threshold:
                        can_merge = True

                if can_merge:
                    # Merge current segment into previous segment
                    prev_segment["end"] = current_segment["end"]
                    prev_segment["text"] = prev_segment["text"] + current_segment["text"]
                    prev_segment["word_scores"] = prev_segment["word_scores"] + current_segment["word_scores"]

                    # Remove the current segment as it's now merged
                    result_segments.pop(i)

                # Move to previous segment
                i -= 1

            # Add all processed segments to the new slice
            new_slice["segments"] = result_segments
            result_slices.append(new_slice)

        return result_slices
    
    def _get_slice_audio_data(self, audio_loader: AudioLoader, slice, slice_length):
        try:
            audio_start_sec = slice["seek"]
            seek_sample = int(audio_start_sec * self._sample_rate)
            slice_length_samples = int(slice_length * self._sample_rate)
            audio_data = audio_loader.next_chunk(seek_sample, slice_length_samples)
            slice_audio_data_as_mp3 = AudioSample(
                audio_data.numpy(), force_read_format="s16le", force_read_sample_rate=self._sample_rate
            ).as_data(no_encode=False, force_out_format="mp3")

            return slice_audio_data_as_mp3
        except Exception as e:
            logger.error(f"Error loading audio for slice seek {float(slice['seek']):.2f}: {e}")
            raise e
    
    def _get_timestamp_token_text(self, seconds: float) -> str:
        """
        Get the timestamp token text for a given seconds.
        This is a helper function to encode the timestamp tokens for the Whisper model.
        It is specific to Whisper and should be moved to a proper util that handles
        timestamp tokens encoding/decoding for any ASR model.
        """
        if 0 <= seconds <= 30:
            # round to precision of .02
            rounded = 0.02 * round(seconds / 0.02)
            return f"<|{rounded:.2f}|>"
        else:
            raise ValueError("Timestamp token out of range.")
    
    def _generate_examples_from_slices(
        self, slices, slice_length, audio_loader, metadata: dict, copy_metadata_fields: list[str] = []
    ) -> Iterator[dict]:
        source_id = metadata.get("source_id", "unknown")
        source_entry_id = metadata.get("source_entry_id", str(uuid.uuid4()))        
        logger.debug(f"Generating dataset from {source_id}/{source_entry_id}")

        # No slices - nothing to do
        if not slices:
            logger.debug(f"No slices in {source_id}/{source_entry_id}")
            return None

        # At least one segments we can work on is expected
        if next(iter([seg for s in slices for seg in s["segments"]]), None) is None:
            logger.debug(f"No segments in {source_id}/{source_entry_id}")
            return None

        prev_example = None
        for slice in slices:
            if slice["segments"]:
                try:
                    slice_text = ""
                    for segment in slice["segments"]:
                        slice_text += self._get_timestamp_token_text(segment["start"])
                        if "text" in segment:
                            slice_text += f'{segment["text"]}{self._get_timestamp_token_text(segment["end"])}'
                    all_word_scores = [score for segment in slice["segments"] for score in segment.get("word_scores", [])]
                    segments_quality_score = self._calculate_median_quality_score(all_word_scores)                    
                    slice_audio_data = self._get_slice_audio_data(audio_loader, slice, slice_length)
                    example = {
                        "audio": {
                            "bytes": slice_audio_data,
                            "path": source_entry_id,
                        },
                        "transcript": slice_text,
                        "metadata": {
                            "seek": float(slice["seek"]),
                            "duration": slice_length,
                            "source": source_id,
                            "entry_id": source_entry_id,
                            "quality_score": segments_quality_score,                            
                        },
                        "has_prev": False,
                        "has_timestamps": True,
                        "prev_transcript": "",
                    }
                    if prev_example:
                        example["prev_transcript"] = prev_example["transcript"]
                        example["has_prev"] = True
                    if copy_metadata_fields:
                        for field_to_copy in copy_metadata_fields:
                            example["metadata"][field_to_copy] = metadata.get(field_to_copy, None)
                    yield example
                    prev_example = example
                except Exception as e:
                    logger.error(
                        f'Error processing slice seek {float(slice["seek"]):.2f} in {source_id}:{source_entry_id}: {e}'
                    )
            else:
                prev_example = None

        logger.debug(f"Done with samples from {source_id}/{source_entry_id}")

    def _is_sample_quality_sufficient(
        self,
        metadata: dict,
        per_sample_quality_threshold: float,
    ) -> bool:
        """Check if the overall sample quality meets the threshold."""
        if per_sample_quality_threshold <= 0:
            return True

        sample_quality_score = metadata.get("quality_score", None)
        if (
            sample_quality_score is not None
            and sample_quality_score < per_sample_quality_threshold
        ):
            logger.debug(
                f"Skipping sample with quality score {sample_quality_score} "
                f"(threshold: {per_sample_quality_threshold})"
            )
            return False
        return True

    def prepare_training_dataset(
        self,
        slice_length: int = 30,
        segments: list[Segment] = None,
        audio_file: str = None,
        per_sample_quality_threshold: float = 0,
        per_segment_quality_threshold: float = 0,
        metadata: dict = None,
        copy_metadata_fields: list[str] = [],
        ) -> Dataset:
        """
        Prepare captioned datasets from the folder.
        Produce audio slices and corresponding text including previous text when available
        Returns a HuggingFace Dataset. Splitting (if needed) should be applied outside this function.
        """

        # Check sample quality before processing
        if not self._is_sample_quality_sufficient(metadata, per_sample_quality_threshold):
            return None

        # Define dataset features
        file_dataset = None
        dataset_features = Features(
            {
                "audio": AudioColumnType(),
                "transcript": ValueColumnType(dtype="string"),
                "metadata": {
                    "seek": ValueColumnType(dtype="float32"),
                    "duration": ValueColumnType(dtype="float32"),
                    "source": ValueColumnType(dtype="string"),
                    "entry_id": ValueColumnType(dtype="string"),
                    "quality_score": ValueColumnType(dtype="float32"), 
                    "source_id": ValueColumnType(dtype="string"),                                   
                },
                "has_prev": ValueColumnType(dtype="bool"),
                "has_timestamps": ValueColumnType(dtype="bool"),
                "prev_transcript": ValueColumnType(dtype="string"),
            }
        )    
        
        # Process each entry individually
        try:
            
            # Load Audio (streams output from an FFMPEG process for memory efficiency)
            audio_loader = AudioLoader(
                str(audio_file),
                stream=True,
                sr=self._sample_rate,
                buffer_size=int(3 * slice_length * self._sample_rate),
            )
            
            try:
                audio_duration = audio_loader.get_duration()

                # Create slices of the captions with the intended slice
                slices = self._generate_slices(segments, audio_duration, slice_length, per_segment_quality_threshold) 
                slices = self._merge_slice_segments(slices)                

                # Collect all examples from this file
                examples = []
                for example in self._generate_examples_from_slices(
                    slices,
                    slice_length,
                    audio_loader,
                    metadata,
                    copy_metadata_fields,
                ):
                    examples.append(example)
                
                # Create dataset from examples if we have any
                if examples:
                    try:
                        file_dataset = Dataset.from_list(examples, features=dataset_features)
                    except Exception as e:
                        logger.error(f"Error creating dataset from {len(examples)} examples: {e}")

            finally:
                try:
                    audio_loader.terminate()
                except Exception as e:
                    logger.error(f"Error terminating audio loader: {e}")
                
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")    
        

        return file_dataset
from abc import ABC, abstractmethod

from datasets import Dataset
from stable_whisper.result import Segment


class DatasetGenerator(ABC):
    @abstractmethod
    def prepare_training_dataset(
        self,
        slice_length: int,
        segments: list[Segment],
        audio_file: str,
        per_sample_quality_threshold: float,
        per_segment_quality_threshold: float,
        metadata: dict,
        copy_metadata_fields: list[str],
    ) -> Dataset | None:
        """Prepare a training dataset from segments and audio."""
        ...

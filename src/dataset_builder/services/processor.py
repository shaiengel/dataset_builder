import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from dataset_builder.config import Config
from dataset_builder.domain.dataset_generator import DatasetGenerator
from dataset_builder.domain.models import Transcript, Vtt
from dataset_builder.domain.parser import Parser
from dataset_builder.domain.segment_result import AlignmentStatus, SegmentResult
from dataset_builder.infrastructure.segment_parser import SegmentParser
from dataset_builder.domain.dataset_manager import DatasetManager
from dataset_builder.services.reader import DatasetReader
from dataset_builder.utils.audio import convert_mp3_to_wav, truncate_wav

logger = logging.getLogger(__name__)


@dataclass
class ProcessedLesson:
    id: str
    transcript: Transcript | None
    vtt: Vtt | None
    segment_result: SegmentResult | None
    wav_path: Path | None
    dataset: Dataset | None


class LessonProcessor:
    def __init__(
        self,
        reader: DatasetReader,
        json_parser: Parser[Transcript],
        vtt_parser: Parser[Vtt],
        segment_parser: SegmentParser,
        dataset_generator: DatasetGenerator,
        dataset_manager: DatasetManager,
        config: Config,
    ) -> None:
        self._reader = reader
        self._json_parser = json_parser
        self._vtt_parser = vtt_parser
        self._segment_parser = segment_parser
        self._dataset_generator = dataset_generator
        self._dataset_manager = dataset_manager
        self._config = config

    def process(self, ids: list[str]) -> list[ProcessedLesson]:
        results = []
        all_datasets: list[Dataset] = []
        logger.info("Starting processing for %d lesson(s): %s", len(ids), ids)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for id in ids:
                logger.info("[%s] Reading files from S3", id)
                data = self._reader.read(id)

                logger.info("[%s] Parsing transcript and VTT", id)
                transcript = self._json_parser.parse(data["json"]) if data["json"] else None
                vtt = self._vtt_parser.parse(data["vtt"]) if data["vtt"] else None

                segment_result = None
                if transcript and vtt:
                    logger.info("[%s] Aligning transcript to VTT", id)
                    segment_result = self._segment_parser.parse(transcript, vtt)
                    logger.info("[%s] Alignment status: %s", id, segment_result.status.value)

                wav_path = None
                if data["audio"]:
                    logger.info("[%s] Converting MP3 to WAV", id)
                    wav_path = convert_mp3_to_wav(data["audio"], tmp_path / f"{id}.wav")

                dataset = None
                if segment_result and wav_path:
                    if segment_result.status == AlignmentStatus.TRUNCATED and segment_result.truncate_at:
                        logger.info("[%s] Truncating WAV at %.3fs", id, segment_result.truncate_at)
                        truncate_wav(wav_path, segment_result.truncate_at)

                    if segment_result.status in (AlignmentStatus.OK, AlignmentStatus.TRUNCATED):
                        logger.info("[%s] Generating training dataset", id)
                        dataset = self._dataset_generator.prepare_training_dataset(
                            slice_length=30,
                            segments=segment_result.segments,
                            audio_file=str(wav_path),
                            per_sample_quality_threshold=0,
                            per_segment_quality_threshold=0,
                            metadata={"source_id": id},
                            copy_metadata_fields=["source_id"],
                        )
                        if dataset:
                            logger.info("[%s] Dataset ready: %d rows", id, len(dataset))
                            all_datasets.append(dataset)

                results.append(
                    ProcessedLesson(
                        id=id,
                        transcript=transcript,
                        vtt=vtt,
                        segment_result=segment_result,
                        wav_path=wav_path,
                        dataset=dataset,
                    )
                )

        if all_datasets:
            combined = self._dataset_manager.concatenate_all_datasets(all_datasets)
            self._dataset_manager.save_to_disk(combined, self._config.output_dataset_path)
        else:
            logger.warning("No datasets produced — nothing saved to disk")

        logger.info("Done processing %d lesson(s)", len(ids))
        return results

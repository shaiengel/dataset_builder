import tempfile
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from dataset_builder.domain.dataset_generator import DatasetGenerator
from dataset_builder.domain.models import Transcript, Vtt
from dataset_builder.domain.parser import Parser
from dataset_builder.domain.segment_result import AlignmentStatus, SegmentResult
from dataset_builder.infrastructure.segment_parser import SegmentParser
from dataset_builder.services.reader import DatasetReader
from dataset_builder.utils.audio import convert_mp3_to_wav, truncate_wav


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
    ) -> None:
        self._reader = reader
        self._json_parser = json_parser
        self._vtt_parser = vtt_parser
        self._segment_parser = segment_parser
        self._dataset_generator = dataset_generator

    def process(self, ids: list[str]) -> list[ProcessedLesson]:
        results = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for id in ids:
                data = self._reader.read(id)
                transcript = self._json_parser.parse(data["json"]) if data["json"] else None
                vtt = self._vtt_parser.parse(data["vtt"]) if data["vtt"] else None

                segment_result = None
                if transcript and vtt:
                    segment_result = self._segment_parser.parse(transcript, vtt)

                wav_path = None
                if data["audio"]:
                    wav_path = convert_mp3_to_wav(data["audio"], tmp_path / f"{id}.wav")

                dataset = None
                if segment_result and wav_path:
                    if segment_result.status == AlignmentStatus.TRUNCATED and segment_result.truncate_at:
                        truncate_wav(wav_path, segment_result.truncate_at)

                    if segment_result.status in (AlignmentStatus.OK, AlignmentStatus.TRUNCATED):
                        dataset = self._dataset_generator.prepare_training_dataset(
                            slice_length=30,
                            segments=segment_result.segments,
                            audio_file=str(wav_path),
                            per_sample_quality_threshold=0,
                            per_segment_quality_threshold=0,
                            metadata={"source_id": id},
                            copy_metadata_fields=["source_id"],
                        )

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
        return results

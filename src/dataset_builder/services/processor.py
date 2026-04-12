import tempfile
from dataclasses import dataclass
from pathlib import Path

from dataset_builder.domain.models import Transcript, Vtt
from dataset_builder.domain.parser import Parser
from dataset_builder.domain.segment_result import SegmentResult
from dataset_builder.infrastructure.segment_parser import SegmentParser
from dataset_builder.services.reader import DatasetReader
from dataset_builder.utils.audio import convert_mp3_to_wav


@dataclass
class ProcessedLesson:
    id: str
    transcript: Transcript | None
    vtt: Vtt | None
    segment_result: SegmentResult | None
    wav_path: Path | None


class LessonProcessor:
    def __init__(
        self,
        reader: DatasetReader,
        json_parser: Parser[Transcript],
        vtt_parser: Parser[Vtt],
        segment_parser: SegmentParser,
    ) -> None:
        self._reader = reader
        self._json_parser = json_parser
        self._vtt_parser = vtt_parser
        self._segment_parser = segment_parser

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

                results.append(
                    ProcessedLesson(
                        id=id,
                        transcript=transcript,
                        vtt=vtt,
                        segment_result=segment_result,
                        wav_path=wav_path,
                    )
                )
        return results

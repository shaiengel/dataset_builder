import json

from dataset_builder.domain.models import Transcript, Word
from dataset_builder.domain.parser import Parser


class JsonParser(Parser[Transcript]):
    def parse(self, content: str) -> Transcript:
        data = json.loads(content)
        words = []
        for seg in data.get("segments", []):
            for w in seg.get("words", []):
                words.append(
                    Word(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        probability=w["probability"],
                    )
                )
        return Transcript(words=words)

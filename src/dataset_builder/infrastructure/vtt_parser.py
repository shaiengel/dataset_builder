import re

from dataset_builder.domain.models import Vtt, VttCue
from dataset_builder.domain.parser import Parser


class VttParser(Parser[Vtt]):
    def parse(self, content: str) -> Vtt:
        cues = []
        pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
        lines = content.strip().split("\n")
        i = 0
        while i < len(lines):
            match = re.match(pattern, lines[i])
            if match:
                start = self._parse_timestamp(match.group(1))
                end = self._parse_timestamp(match.group(2))
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(pattern, lines[i]):
                    text_lines.append(lines[i])
                    i += 1
                cues.append(VttCue(start=start, end=end, text="\n".join(text_lines)))
            else:
                i += 1
        return Vtt(cues=cues)

    def _parse_timestamp(self, ts: str) -> float:
        h, m, rest = ts.split(":")
        s, ms = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

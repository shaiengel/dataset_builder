import logging

from dataset_builder.config import Config
from dataset_builder.domain.file_manager import FileManager

logger = logging.getLogger(__name__)


class DatasetReader:
    def __init__(self, s3_client: FileManager, config: Config) -> None:
        self._s3 = s3_client
        self._config = config

    def list_ids(self) -> list[str]:
        keys = self._s3.list_keys(self._config.audio_bucket, suffix=".mp3")
        return [key.removesuffix(".mp3") for key in keys]

    def read(self, id: str) -> dict | None:
        json = self._s3.get_text(self._config.transcription_bucket, f"{id}.json")
        vtt = self._s3.get_text(self._config.transcription_bucket, f"{id}.vtt")
        audio = self._s3.get_bytes(self._config.audio_bucket, f"{id}.mp3")

        if json is None or vtt is None or audio is None:
            missing = [name for name, val in (("json", json), ("vtt", vtt), ("audio", audio)) if val is None]
            logger.warning("[%s] Skipping — failed to read: %s", id, ", ".join(missing))
            return None

        return {"json": json, "vtt": vtt, "audio": audio}

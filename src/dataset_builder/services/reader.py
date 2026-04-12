from dataset_builder.config import Config
from dataset_builder.domain.file_manager import FileManager


class DatasetReader:
    def __init__(self, s3_client: FileManager, config: Config) -> None:
        self._s3 = s3_client
        self._config = config

    def read(self, id: str) -> dict:
        return {
            "json": self._s3.get_text(self._config.transcription_bucket, f"{id}.json"),
            "vtt": self._s3.get_text(self._config.transcription_bucket, f"{id}.vtt"),
            "audio": self._s3.get_bytes(self._config.audio_bucket, f"{id}.mp3"),
        }

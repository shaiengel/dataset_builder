import logging
from pathlib import Path

import datasets
from datasets import Dataset

from dataset_builder.config import Config
from dataset_builder.domain.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class EvaluationProcessor:
    def __init__(self, dataset_manager: DatasetManager, config: Config) -> None:
        self._dataset_manager = dataset_manager
        self._config = config

    def process(self, evaluation_dir: Path, entries: list[dict]) -> Dataset | None:
        features = datasets.Features({
            "audio": datasets.Audio(sampling_rate=16000),
            "transcript": datasets.Value("string"),
            "metadata": {
                "id": datasets.Value("int64"),
                "name": datasets.Value("string"),
                "daf": datasets.Value("string"),
            },
        })

        rows = []
        for entry in entries:
            id_ = entry["id"]
            mp3_path = evaluation_dir / f"{id_}.mp3"
            txt_path = evaluation_dir / f"{id_}.txt"

            if not mp3_path.exists():
                logger.warning("[%s] Missing MP3, skipping", id_)
                continue
            if not txt_path.exists():
                logger.warning("[%s] Missing transcript, skipping", id_)
                continue

            transcript = txt_path.read_text(encoding="utf-8").strip()
            rows.append({
                "audio": str(mp3_path),
                "transcript": transcript,
                "metadata": {"id": id_, "name": entry["name"], "daf": entry["daf"]},
            })
            logger.info("[%s] Added to evaluation dataset", id_)

        if not rows:
            logger.warning("No evaluation rows produced — nothing saved")
            return None

        dataset = Dataset.from_list(rows, features=features)
        self._dataset_manager.save_to_disk(dataset, self._config.output_evaluation_path)
        logger.info("Saved evaluation dataset (%d rows) to %s", len(dataset), self._config.output_evaluation_path)
        return dataset

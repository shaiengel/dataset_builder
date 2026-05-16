import json
import logging
import sys
from pathlib import Path

from dataset_builder.infrastructure.dependency_injection import DependenciesContainer
from dataset_builder.services.evaluation_processor import EvaluationProcessor

EVALUATION_DIR = Path("evaluation")
DATA_JSON = Path("eval_data.json")

sys.stderr.reconfigure(encoding="utf-8")

_log_format = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_format)
_file_handler = logging.FileHandler("eval_dataset_builder.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_log_format))
logging.getLogger().addHandler(_file_handler)


def main():
    entries = json.loads(DATA_JSON.read_text(encoding="utf-8"))

    container = DependenciesContainer()
    processor = EvaluationProcessor(
        dataset_manager=container.dataset_manager(),
        config=container.config(),
    )
    dataset = processor.process(EVALUATION_DIR, entries)

    if dataset:
        print(f"\n=== Evaluation Dataset Summary ===")
        print(f"  rows : {len(dataset)}")
        print(f"  saved to : {container.config().output_evaluation_path}")
    else:
        print("No evaluation dataset produced.")


if __name__ == "__main__":
    main()

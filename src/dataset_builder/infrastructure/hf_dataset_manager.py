import argparse
import logging

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from huggingface_hub import DatasetCard, DatasetCardData

from dataset_builder.config import Config
from dataset_builder.domain.dataset_manager import DatasetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = Config()

class HuggingFaceDatasetManager(DatasetManager):

    def concatenate_all_datasets(self, datasets: list[Dataset]) -> Dataset:
        logger.info("Concatenating %d dataset(s)", len(datasets))
        combined = concatenate_datasets(datasets)
        logger.info("Combined dataset: %d rows", len(combined))
        return combined

    def save_to_disk(self, dataset: Dataset | DatasetDict, path: str) -> None:
        logger.info("Saving dataset to %s", path)
        dataset.save_to_disk(path)
        logger.info("Dataset saved")

    def load_dataset_from_disk(self, path: str) -> Dataset | DatasetDict:
        logger.info("Loading dataset from %s", path)
        return load_from_disk(path)

    def split_dataset(self, dataset: Dataset, test_size: float = 0.1) -> DatasetDict:
        logger.info("Splitting dataset (test_size=%.2f)", test_size)
        split = dataset.train_test_split(test_size=test_size)
        logger.info("Train: %d rows, Test: %d rows", len(split["train"]), len(split["test"]))
        return split

    def upload_dataset_to_hub(self, dataset: Dataset | DatasetDict, repo_id: str, max_shard_size: str = "500MB", token: str | None = None) -> None:
        logger.info("Uploading dataset to hub: %s", repo_id)
        dataset.push_to_hub(repo_id, max_shard_size=max_shard_size, token=token)
        logger.info("Upload complete")

    def create_dataset_card(
        self,
        language: str,
        license: str,
        description: str,
        pretty_name: str,
        template_path: str | None = None,
    ) -> DatasetCard:
        card_data = DatasetCardData(
            language=language,
            license=license,
            dataset_description=description,
            pretty_name=pretty_name,
        )
        return DatasetCard.from_template(card_data, template_path=template_path)

    def upload_dataset_card_to_hub(self, card: DatasetCard, repo_id: str, token: str | None = None) -> None:
        logger.info("Uploading dataset card to hub: %s", repo_id)
        card.push_to_hub(repo_id=repo_id, repo_type="dataset", token=token)
        logger.info("Dataset card upload complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, split, and upload dataset to HuggingFace Hub")
    parser.add_argument("--dataset-path", default=config.output_dataset_path, help="Path to dataset on disk")
    parser.add_argument("--repo-id", required=True, help="HuggingFace Hub repo ID (e.g. org/dataset-name)")
    parser.add_argument("--test-size", type=float, default=config.test_size, help="Fraction of data for test split")
    parser.add_argument("--language", default=config.card_language, help="Dataset language code")
    parser.add_argument("--license", default=config.card_license, help="Dataset license")
    parser.add_argument("--description", default=config.card_description, help="Dataset description")
    parser.add_argument("--pretty-name", default=config.card_pretty_name, help="Dataset pretty name")
    parser.add_argument("--template-path", default=config.card_template_path, help="Path to dataset card template")
    args = parser.parse_args()

    token = config.hf_token
    manager = HuggingFaceDatasetManager()
    dataset = manager.load_dataset_from_disk(args.dataset_path)
    split = manager.split_dataset(dataset, test_size=args.test_size)
    manager.upload_dataset_to_hub(split, args.repo_id, max_shard_size=config.max_shard_size, token=token)
    card = manager.create_dataset_card(
        language=args.language,
        license=args.license,
        description=args.description,
        pretty_name=args.pretty_name,
        template_path=args.template_path,
    )
    manager.upload_dataset_card_to_hub(card, args.repo_id, token=token)

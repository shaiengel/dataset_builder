import logging

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from dataset_builder.domain.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


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

    def upload_dataset_to_hub(self, dataset: Dataset | DatasetDict, repo_id: str) -> None:
        logger.info("Uploading dataset to hub: %s", repo_id)
        dataset.push_to_hub(repo_id, max_shard_size="500MB")
        logger.info("Upload complete")

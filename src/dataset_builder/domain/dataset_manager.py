from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard


class DatasetManager(ABC):

    @abstractmethod
    def concatenate_all_datasets(self, datasets: list[Dataset]) -> Dataset: ...

    @abstractmethod
    def save_to_disk(self, dataset: Dataset | DatasetDict, path: str) -> None: ...

    @abstractmethod
    def load_dataset_from_disk(self, path: str) -> Dataset | DatasetDict: ...

    @abstractmethod
    def split_dataset(self, dataset: Dataset, test_size: float = 0.1) -> DatasetDict: ...

    @abstractmethod
    def upload_dataset_to_hub(self, dataset: Dataset | DatasetDict, repo_id: str, max_shard_size: str = "500MB") -> None: ...

    @abstractmethod
    def create_dataset_card(
        self,
        language: str,
        license: str,
        description: str,
        pretty_name: str,
        template_path: str | None = None,
    ) -> DatasetCard: ...

    @abstractmethod
    def upload_dataset_card_to_hub(self, card: DatasetCard, repo_id: str) -> None: ...

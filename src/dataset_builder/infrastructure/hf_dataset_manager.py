import argparse
import logging
import random
import tempfile
import uuid
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from huggingface_hub import DatasetCard, DatasetCardData, HfApi

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

    def split_dataset(self, dataset: Dataset, test_size: float = 0.1, group_by: str | None = "source_id", seed: int = 42) -> DatasetDict:
        if group_by is None:
            logger.info("Splitting dataset randomly (test_size=%.2f)", test_size)
            split = dataset.train_test_split(test_size=test_size, seed=seed)
        else:
            logger.info("Splitting dataset by metadata.%s (test_size=%.2f)", group_by, test_size)
            split = self._split_by_group(dataset, test_size, group_by, seed)
        logger.info("Train: %d rows, Test: %d rows", len(split["train"]), len(split["test"]))
        return split

    def _split_by_group(self, dataset: Dataset, test_size: float, group_by: str, seed: int) -> DatasetDict:
        # Whole groups (lessons) go to one side, so adjacent slices never straddle train/test.
        metadata = dataset.select_columns(["metadata"])["metadata"]
        groups: dict[str, list[int]] = {}
        for idx, meta in enumerate(metadata):
            groups.setdefault(meta[group_by], []).append(idx)

        # Only all-digit ids are eligible for test; ids containing letters always stay in train.
        group_ids = sorted(gid for gid in groups if gid.isdigit())
        random.Random(seed).shuffle(group_ids)

        target_rows = int(len(dataset) * test_size)
        test_idx: list[int] = []
        test_group_ids: list[str] = []
        for gid in group_ids:
            if len(test_idx) >= target_rows:
                break
            test_idx.extend(groups[gid])
            test_group_ids.append(gid)
        logger.info(
            "Test split: %d of %d groups (%d numeric-eligible)",
            len(test_group_ids), len(groups), len(group_ids),
        )
        logger.info("Test %s values: %s", group_by, ", ".join(sorted(test_group_ids, key=int)))

        test_set = set(test_idx)
        train_idx = [i for i in range(len(dataset)) if i not in test_set]
        # train_test_split shuffles by default; keep that so rows are not stored in lesson order.
        rng = random.Random(seed)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        return DatasetDict({
            "train": dataset.select(train_idx),
            "test": dataset.select(test_idx),
        })

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

    def append_to_hub(
        self,
        dataset: Dataset,
        repo_id: str,
        test_size: float = 0.1,
        max_shard_size: str = "500MB",
        token: str | None = None,
        group_by: str | None = "source_id",
    ) -> None:
        logger.info("Appending dataset (%d rows) to hub: %s", len(dataset), repo_id)
        split = self.split_dataset(dataset, test_size=test_size, group_by=group_by)
        batch_id = uuid.uuid4().hex[:8]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "data"
            data_dir.mkdir()

            for split_name in ("train", "test"):
                split_dataset = split[split_name]
                split_dataset.to_parquet(data_dir / f"{split_name}-00000-of-00001-{batch_id}.parquet")
                logger.info(
                    "Saved %s split: %d rows as %s-00000-of-00001-%s.parquet",
                    split_name, len(split_dataset), split_name, batch_id,
                )

            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(data_dir),
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Append batch {batch_id}: {len(dataset)} rows",
            )
        logger.info("Append complete — batch %s uploaded", batch_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, split, and upload dataset to HuggingFace Hub")
    parser.add_argument("--dataset-path", default=config.output_dataset_path, help="Path to dataset on disk")
    parser.add_argument("--repo-id", required=True, help="HuggingFace Hub repo ID (e.g. org/dataset-name)")
    parser.add_argument("--test-size", type=float, default=config.test_size, help="Fraction of data for test split")
    parser.add_argument("--group-by", default="source_id", help="metadata field to keep whole groups in one split (lesson id)")
    parser.add_argument("--random-split", action="store_true", help="Split rows randomly, ignoring --group-by")
    parser.add_argument("--split-output-path", default="output/dataset_split", help="Where to save the split dataset before upload (re-upload with --dataset-path)")
    parser.add_argument("--language", default=config.card_language, help="Dataset language code")
    parser.add_argument("--license", default=config.card_license, help="Dataset license")
    parser.add_argument("--description", default=config.card_description, help="Dataset description")
    parser.add_argument("--pretty-name", default=config.card_pretty_name, help="Dataset pretty name")
    parser.add_argument("--template-path", default=config.card_template_path, help="Path to dataset card template")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--no-split", action="store_true", help="Upload dataset without train/test split")
    mode_group.add_argument("--append", action="store_true", help="Append to existing HF dataset (upload new parquet shards)")

    args = parser.parse_args()

    token = config.hf_token
    manager = HuggingFaceDatasetManager()
    dataset = manager.load_dataset_from_disk(args.dataset_path)

    group_by = None if args.random_split else args.group_by

    if args.append:
        manager.append_to_hub(dataset, args.repo_id, test_size=args.test_size, max_shard_size=config.max_shard_size, token=token, group_by=group_by)
    elif args.no_split:
        manager.upload_dataset_to_hub(dataset, args.repo_id, max_shard_size=config.max_shard_size, token=token)
    else:
        if isinstance(dataset, DatasetDict):
            logger.info("Dataset on disk is already split, skipping split")
            split = dataset
        else:
            split = manager.split_dataset(dataset, test_size=args.test_size, group_by=group_by)
            # Persist the split first so a failed upload can be retried without re-splitting.
            manager.save_to_disk(split, args.split_output_path)
        manager.upload_dataset_to_hub(split, args.repo_id, max_shard_size=config.max_shard_size, token=token)

    if not args.append:
        card = manager.create_dataset_card(
            language=args.language,
            license=args.license,
            description=args.description,
            pretty_name=args.pretty_name,
            template_path=args.template_path,
        )
        manager.upload_dataset_card_to_hub(card, args.repo_id, token=token)

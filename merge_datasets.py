"""Merge multiple local Arrow dataset folders into one."""

import argparse
import logging

from datasets import concatenate_datasets, load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple Arrow dataset folders into one")
    parser.add_argument("inputs", nargs="+", help="Paths to dataset folders to merge")
    parser.add_argument("--output", required=True, help="Output path for merged dataset")
    args = parser.parse_args()

    datasets = []
    for path in args.inputs:
        ds = load_from_disk(path)
        logger.info("Loaded %s: %d rows", path, len(ds))
        datasets.append(ds)

    merged = concatenate_datasets(datasets)
    logger.info("Merged: %d rows total", len(merged))

    merged.save_to_disk(args.output)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()

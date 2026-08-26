import json
from pathlib import Path


def _load_ids_from_file(file: Path) -> set[str]:
    if not file.exists():
        return set()
    data = json.loads(file.read_text())
    return set(data) if isinstance(data, list) else {lid for dataset in data.get("datasets", []) for lid in dataset.get("list_ids", [])}


def filter_new_ids(ids: list[str], progress_file: Path, failed_file: Path | None = None) -> list[str]:
    seen = _load_ids_from_file(progress_file)
    if failed_file:
        seen |= _load_ids_from_file(failed_file)
    return [i for i in ids if i not in seen]


def save_failed_ids(ids: list[str], failed_file: Path) -> None:
    existing = _load_ids_from_file(failed_file)
    existing.update(ids)
    failed_file.parent.mkdir(parents=True, exist_ok=True)
    failed_file.write_text(json.dumps(sorted(existing), indent=2))


def save_progress(ids: list[str], duration: int, progress_file: Path) -> None:
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
    else:
        data = {}

    data.setdefault("datasets", [])
    data.setdefault("total_duration", 0)

    existing_ids = {d["dataset_id"] for d in data["datasets"]}
    next_id = max(existing_ids) + 1 if existing_ids else 1

    data["datasets"].append({"dataset_id": next_id, "list_ids": ids, "duration": duration})
    data["total_duration"] += duration

    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(data, indent=2))

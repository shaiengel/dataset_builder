import json
from pathlib import Path


def filter_new_ids(ids: list[str], progress_file: Path) -> list[str]:
    if not progress_file.exists():
        return ids
    data = json.loads(progress_file.read_text())
    seen = {lid for dataset in data.get("datasets", []) for lid in dataset.get("list_ids", [])}
    return [i for i in ids if i not in seen]


def save_progress(ids: list[str], duration: int, progress_file: Path) -> None:
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
    else:
        data = {"datasets": [], "total_duration": 0}

    existing_ids = {d["dataset_id"] for d in data["datasets"]}
    next_id = max(existing_ids) + 1 if existing_ids else 1

    data["datasets"].append({"dataset_id": next_id, "list_ids": ids, "duration": duration})
    data["total_duration"] += duration

    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(data, indent=2))

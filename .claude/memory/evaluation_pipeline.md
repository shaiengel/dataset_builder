# Evaluation Pipeline

## Overview

A separate pipeline from the training dataset builder that produces a HuggingFace dataset for WER evaluation. Input is local files in `evaluation/`; output is a saved HF dataset at `output/evaluation`.

## Entry Point

`eval_main.py` (project root) — reads `evaluation/data.json`, wires `EvaluationProcessor` via the existing `DependenciesContainer`, prints a summary.

Run with: `uv run python eval_main.py`

## Input

`evaluation/` folder contains:
- `data.json` — list of entries with `id` (int), `name`, `daf` fields
- `{id}.mp3` — full-lesson audio
- `{id}.txt` — plain-text reference transcript

Current entries: 6 lessons of `menachot_106` (yehuda_eliyahu, haim_shmarler, mazlich_chay_mazuz, hadaf_beyiyun, yehezkel_tzishngel, efraim_segal).

## Service

`src/dataset_builder/services/evaluation_processor.py` — `EvaluationProcessor`

- Constructor: `(dataset_manager: DatasetManager, config: Config)`
- `process(evaluation_dir: Path, entries: list[dict]) -> Dataset | None`
  - Reads MP3 path + `.txt` transcript per entry
  - Builds HF `Dataset` with 3 columns (see below)
  - Calls `dataset_manager.save_to_disk(dataset, config.output_evaluation_path)`

## Dataset Schema

| Column | HF Feature | Content |
|--------|-----------|---------|
| `audio` | `datasets.Audio(sampling_rate=16000)` | MP3 file path; decoded lazily to `{"array": float32, "sampling_rate": 16000}` |
| `transcript` | `datasets.Value("string")` | Plain-text reference transcript |
| `metadata` | nested dict | `{"id": int64, "name": str, "daf": str}` from `data.json` |

## Audio Format — No Conversion Needed

All ASR engines in `C:\portal\asr-training-1\engines\` expect `{"array": float32_ndarray, "sampling_rate": int}`. `datasets.Audio` decodes the MP3 to that format automatically on access — no WAV conversion or `AudioSample` processing required (unlike the training pipeline which slices audio into 30s chunks via `AudioLoader`).

## Config

- Env var: `OUTPUT_EVALUATION_PATH=output/evaluation` (in `.env`)
- Config field: `Config.output_evaluation_path`

## VS Code

Launch config: `"Python: eval_main.py"` in `.vscode/launch.json`.

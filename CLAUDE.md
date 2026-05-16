# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dataset-builder` has two pipelines:

1. **Training pipeline** (`main.py`) — reads lesson files from S3 (JSON transcript, VTT subtitles, MP3 audio), aligns them, and produces a HuggingFace dataset of 30-second audio slices with Whisper-style timestamp tokens for fine-tuning.
2. **Evaluation pipeline** (`eval_main.py`) — reads local MP3 and plain-text transcripts from `evaluation/`, and produces a HuggingFace dataset with full-lesson audio, reference transcripts, and metadata for WER evaluation.

## Prerequisites

### ffmpeg (Windows)

ffmpeg is required for MP3-to-WAV conversion (training pipeline). Install via winget:

```powershell
winget install --id Gyan.FFmpeg
```

Then restart your terminal so `ffmpeg` is on `PATH`.

## Commands

```bash
# Build training dataset
uv run python main.py

# Build evaluation dataset
uv run python eval_main.py

# Add a dependency
uv add <package>

# Sync the environment
uv sync
```

## Structure

```
eval_data.json                     # Evaluation entries (id, name, daf)
evaluation_card_template.md        # HF dataset card template for evaluation dataset
dataset_card_template.md           # HF dataset card template for training dataset
evaluation/                        # Local eval audio + transcripts ({id}.mp3, {id}.txt)

src/dataset_builder/
├── domain/                        # Domain models and ABCs
│   ├── file_manager.py            # FileManager ABC
│   ├── dataset_manager.py         # DatasetManager ABC
│   ├── dataset_generator.py       # DatasetGenerator ABC
│   ├── models.py                  # Word, Transcript, VttCue, Vtt
│   ├── parser.py                  # Parser[T] ABC
│   └── segment_result.py          # AlignmentStatus, SegmentResult
├── infrastructure/                # Implementations
│   ├── s3_client.py               # S3Client (FileManager impl)
│   ├── json_parser.py             # JsonParser (stable_whisper JSON → Transcript)
│   ├── vtt_parser.py              # VttParser (VTT → Vtt)
│   ├── segment_parser.py          # SegmentParser (alignment + Segment building)
│   ├── whisper_dataset_generator.py # WhisperDatasetGenerator (30s slices)
│   ├── hf_dataset_manager.py      # HuggingFaceDatasetManager + upload CLI
│   └── dependency_injection.py    # DI container
├── services/                      # Business logic
│   ├── reader.py                  # DatasetReader (S3 file fetching)
│   ├── processor.py               # LessonProcessor (training orchestration)
│   ├── evaluation_processor.py    # EvaluationProcessor (eval dataset building)
│   └── progress_tracker.py        # filter_new_ids / save_progress
└── config.py                      # Config dataclass
```

## Key Concepts

- **Transcript**: Flattened list of words from stable_whisper JSON (word, start, end, probability)
- **Vtt**: List of VttCues (start, end, text, duration)
- **SegmentResult**: Aligned segments with status (OK/TRUNCATED/NO_DATA) and truncate_at timestamp
- **Segment**: `stable_whisper.result.Segment` with `WordTiming` objects

## S3 Buckets (training pipeline only)

- `final-transcription` — JSON and VTT files (`{id}.json`, `{id}.vtt`)
- `portal-daf-yomi-audio` — MP3 files (`{id}.mp3`)

## Output Paths

- `output/dataset` — training dataset (Arrow format)
- `output/evaluation` — evaluation dataset (Arrow format)

## HuggingFace Upload

`hf_dataset_manager.py` doubles as a CLI for uploading datasets to the Hub:

```bash
# Upload training dataset (with train/test split)
uv run python -m dataset_builder.infrastructure.hf_dataset_manager \
  --repo-id portal-daf-yomi/portal-daf-yomi-whisper-training --test-size 0.1

# Upload evaluation dataset (no split)
uv run python -m dataset_builder.infrastructure.hf_dataset_manager \
  --dataset-path output/evaluation \
  --repo-id portal-daf-yomi/portal-daf-yomi-evaluation \
  --no-split --template-path evaluation_card_template.md
```

VS Code launch configs for both are in `.vscode/launch.json`.

## Patterns

- Dependency injection via `dependency-injector`
- Abstract base classes in domain, implementations in infrastructure
- AWS profile defaults to `portal`

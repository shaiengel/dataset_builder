---
name: Dataset Builder Project Overview
description: Architecture and purpose of the dataset-builder repo for building whisper training datasets from S3 audio/transcripts
type: project
---

# Dataset Builder

## Purpose

Builds training datasets for Whisper fine-tuning by:
1. Reading lesson files from S3 (JSON transcript, VTT subtitles, MP3 audio)
2. Parsing and aligning JSON (stable_whisper output) with VTT
3. Producing `stable_whisper.result.Segment` objects with word-level timing and probability

## Architecture

Uses dependency injection (`dependency-injector`) with clean separation:

### Domain Layer (`src/dataset_builder/domain/`)
- `file_manager.py` — `FileManager` ABC for file operations
- `models.py` — `Word`, `Transcript`, `VttCue`, `Vtt` dataclasses
- `parser.py` — generic `Parser[T]` ABC
- `segment_result.py` — `AlignmentStatus` enum, `SegmentResult` with truncation info

### Infrastructure Layer (`src/dataset_builder/infrastructure/`)
- `s3_client.py` — `S3Client` implements `FileManager` using boto3
- `json_parser.py` — parses stable_whisper JSON into `Transcript` (flattened word list)
- `vtt_parser.py` — parses VTT into `Vtt` with cues
- `segment_parser.py` — aligns JSON+VTT, produces `Segment` objects with `WordTiming`
- `dependency_injection.py` — `DependenciesContainer` wiring all components

### Services Layer (`src/dataset_builder/services/`)
- `reader.py` — `DatasetReader` fetches files from S3 buckets
- `processor.py` — `LessonProcessor` orchestrates reading, parsing, alignment

## Key Data Flow

```
S3 Buckets → DatasetReader.read(id)
    ↓
JSON string → JsonParser → Transcript (list of Words with timestamps/probabilities)
VTT string  → VttParser  → Vtt (list of VttCues)
    ↓
SegmentParser.parse(transcript, vtt)
    ↓
SegmentResult (list of stable_whisper Segments, alignment status, truncate_at timestamp)
```

## Alignment Behavior

- Matches words from JSON with VTT word-by-word
- On mismatch: stops, returns `TRUNCATED` status with `truncate_at` timestamp
- If VTT shorter: truncates to VTT length
- Returns `OK` if fully aligned

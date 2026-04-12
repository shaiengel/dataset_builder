# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dataset-builder` builds training datasets for Whisper fine-tuning by reading lesson files from S3 (JSON transcript, VTT subtitles, MP3 audio), aligning them, and producing `stable_whisper.result.Segment` objects.

## Commands

```bash
# Run the application
uv run python main.py

# Add a dependency
uv add <package>

# Sync the environment
uv sync
```

## Structure

```
src/dataset_builder/
├── domain/                    # Domain models and ABCs
│   ├── file_manager.py        # FileManager ABC
│   ├── models.py              # Word, Transcript, VttCue, Vtt
│   ├── parser.py              # Parser[T] ABC
│   └── segment_result.py      # AlignmentStatus, SegmentResult
├── infrastructure/            # Implementations
│   ├── s3_client.py           # S3Client (FileManager impl)
│   ├── json_parser.py         # JsonParser (stable_whisper JSON → Transcript)
│   ├── vtt_parser.py          # VttParser (VTT → Vtt)
│   ├── segment_parser.py      # SegmentParser (alignment + Segment building)
│   └── dependency_injection.py # DI container
├── services/                  # Business logic
│   ├── reader.py              # DatasetReader (S3 file fetching)
│   └── processor.py           # LessonProcessor (orchestration)
└── config.py                  # Config dataclass
```

## Key Concepts

- **Transcript**: Flattened list of words from stable_whisper JSON (word, start, end, probability)
- **Vtt**: List of VttCues (start, end, text, duration)
- **SegmentResult**: Aligned segments with status (OK/TRUNCATED/NO_DATA) and truncate_at timestamp
- **Segment**: `stable_whisper.result.Segment` with `WordTiming` objects

## S3 Buckets

- `final-transcription` — JSON and VTT files (`{id}.json`, `{id}.vtt`)
- `portal-daf-yomi-audio` — MP3 files (`{id}.mp3`)

## Patterns

- Dependency injection via `dependency-injector`
- Abstract base classes in domain, implementations in infrastructure
- AWS profile defaults to `portal`

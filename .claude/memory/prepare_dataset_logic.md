---
name: Prepare Dataset Logic
description: How WhisperDatasetGenerator prepares training datasets with slicing, quality filtering, and timestamp tokens
type: project
---

# Prepare Dataset Logic

## Overview

`WhisperDatasetGenerator.prepare_training_dataset()` converts aligned segments + audio into a HuggingFace Dataset for Whisper fine-tuning.

## Pipeline

```
segments + audio_file
        ↓
_is_sample_quality_sufficient()  → skip if quality_score from metadata < threshold
        ↓
_generate_slices()               → split into 30s slices with quality control
        ↓
_merge_slice_segments()          → merge segments with small gaps (<0.3s)
        ↓
_generate_examples_from_slices() → create dataset records with audio + transcript
        ↓
Dataset.from_list()
```

## Slicing Logic (`_generate_slices`)

Divides audio into fixed-length slices (default 30s) while respecting segment boundaries.

### Key Concepts

- **Slice**: A fixed-length audio chunk (e.g., 30s) starting at `seek` position
- **Segment**: A transcript unit with start/end times and text (e.g., a sentence)
- **Slice segment**: A segment's data relative to slice start (times are 0-based within slice)
- **Crossed-over segment**: A segment that starts in one slice but ends beyond it

### Algorithm Overview

```
while not at end of audio:
    1. Skip invalid segments (too long or zero duration)
    2. Create new slice at current position
    3. Add segments that START within slice
    4. Handle segment overflow (segment crosses slice boundary)
    5. Quality check → abandon low-quality slices
```

### Step-by-Step Logic

#### 1. Skip Invalid Segments
```
If current segment duration > slice_length OR duration == 0:
    → Skip to next segment
    → Continue from next segment's start time
```
Segments longer than a slice can't fit anywhere; zero-duration segments are useless.

#### 2. Create Slice
```python
slice = {
    "seek": slice_start,      # absolute position in audio
    "segments": []            # will hold segment data
}
slice_end = slice_start + slice_length  # or audio_duration if near end
```
Default: `next_slice_start = slice_start + slice_length` (may be overridden by overflow handling).

#### 3. Add Segments

For each segment where `segment.start < slice_end`:

**If segment ENDS within slice** (complete segment):
```python
slice_segment = {
    "start": segment.start - slice_start,  # relative to slice
    "end": segment.end - slice_start,      # relative to slice
    "text": segment.text,
    "word_scores": [word probabilities]
}
```
Advance to next segment.

**If segment ENDS beyond slice** (crossed-over):
Only add `{"start": relative_start}` (no end/text) as a marker.
Then handle overflow (see below).

#### 4. Handle Segment Overflow

When a segment **starts** in this slice but **ends** beyond it:

**Case A: Crossed-over segment is the ONLY one in slice**
```
           s                    e
          /                     \
|_________________________||........
         ↑ slice boundary
```
- Clear the slice (leave it empty)
- Start next slice at this segment's START
- Don't advance segment index (will retry this segment)

**Case B: Crossed-over follows completed segments**
```
  s    e  s                    e
 /     \ /                     \
|_________________________||........
        ↑ slice boundary
```
- Keep the completed segments + the start-only marker
- Start next slice at PREVIOUS (completed) segment's END
- Don't advance segment index (will retry crossed-over segment)

**Note on crossed-over markers**: The `{"start": X}` entry (no end/text) marks where a crossed-over segment begins. This is kept in the slice output but the segment itself is NOT included in quality scoring.

#### 5. Quality Control

After building a slice, calculate quality from **completed segments only** (not crossed-over):
```
If slice quality < threshold:
    1. Search for first good-quality segment (skip index 0 to avoid infinite loop)
    2. If found: start next slice from that segment's start
    3. If not found: skip to segment AFTER the last one in slice
    4. Clear current slice segments (abandon it, but slice entry remains with empty segments)
```

### Edge Cases (from tests)

| Case | Behavior |
|------|----------|
| Zero-duration segment | Skipped entirely |
| Segment longer than slice_length | Skipped entirely |
| Audio duration < slice_length | Single slice, clips at audio end |
| Zero audio duration | Returns empty list |
| First slice has no segments | Empty slice at seek=0 |
| Last slice has no segments | Empty slice remains |
| Crossed-over twice | Each retry creates new slice at segment start |
| All segments low quality | Multiple empty slices, then good segment's slice |

### Output Structure

```python
slices = [
    {
        "seek": 0,
        "segments": [
            {"start": 5, "end": 10, "text": "Hello", "word_scores": [0.8]},
            {"start": 20, "end": 30, "text": "World", "word_scores": [0.9]},
        ]
    },
    {
        "seek": 30,
        "segments": []  # empty slice (gap or quality filtered)
    },
    {
        "seek": 36,
        "segments": [{"start": 0, "end": 1, "text": "high", "word_scores": [0.9]}]
    },
]
```

### Visual Examples

**Example 1: Normal cross-over**
```
Segments:  [Hello]     [World------>]
           0-10        15-35
Slice length: 30, Audio: 40

Slice 1 (seek=0):  [Hello: 0-10] [World: start=15 only]
                   → World crosses boundary, next slice at Hello.end (10)
Slice 2 (seek=10): [World: 5-25]
                   → World now fits (relative times)
```

**Example 2: Crossed-over is only segment**
```
Segments:  [Hello]     [World------------->]
           2-4         29-35
Slice length: 30, Audio: 40

Slice 1 (seek=0):  [Hello: 2-4] [World: start=29 only]
                   → World crosses, next slice at Hello.end (4)
Slice 2 (seek=4):  empty (World still doesn't fit: starts at 25 relative, ends at 31)
                   → World is only segment, next slice at World.start (29)
Slice 3 (seek=29): [World: 0-6]
                   → World finally fits
```

**Example 3: Quality filtering**
```
Segments:  [high]  [low] [low] [low] [high]
           5-10    15-20 20-21 21-22 25-30
Quality:   0.8     0.1   0.1   0.1   0.8
Threshold: 0.8

Slice 1 (seek=0):  median([0.8,0.1,0.1,0.1]) = 0.1 < 0.8
                   → Search for good segment, skip first
                   → None found above threshold
                   → Skip to segment after last (index 4)
                   → Clear slice → empty
Slice 2 (seek=25): [high: 0-5]
                   → Quality OK
```

## Merge Logic (`_merge_slice_segments`)

Merges adjacent segments within each slice when their gap is below a threshold (default 0.3s).

### Purpose

After slicing, segments may be artificially separated by small gaps. Merging them:
- Creates more natural transcript units
- Reduces the number of timestamp tokens
- Improves training data quality

### Algorithm

```
For each slice:
    Process segments in REVERSE order (right to left):
        If current segment is complete (has start, end, text):
            Calculate gap = current.start - previous.end
            If gap < threshold:
                Merge current INTO previous:
                    - previous.end = current.end
                    - previous.text = previous.text + current.text
                    - previous.word_scores = previous.word_scores + current.word_scores
                Remove current segment
```

### Why Reverse Order?

Processing right-to-left allows merging chains without index shifting issues:
```
Segments: [A] [B] [C] [D]  (all close together)

Right-to-left:
  D merges into C → [A] [B] [CD]
  CD merges into B → [A] [BCD]
  BCD merges into A → [ABCD]

Left-to-right would require re-checking after each merge.
```

### Merge Conditions

A segment can be merged into the previous one if:
1. It has `start`, `end`, AND `text` keys (complete segment)
2. Gap between them is < `merge_below_gap_threshold` (default 0.3s)

**Incomplete segments** (crossed-over markers with only `start`) are **never merged** and act as merge barriers.

### Edge Cases (from tests)

| Case | Behavior |
|------|----------|
| Empty slices list | Returns empty list |
| Slice with no segments | Returns unchanged |
| Single segment | Returns unchanged |
| Gap >= threshold | No merge |
| Gap < threshold | Merge |
| Incomplete segment (no end/text) | Not merged, blocks chain |
| Multiple slices | Each processed independently |

### Examples

**Example 1: Simple merge**
```
Input:  [Hello: 0-5] [world: 5.2-10]
Gap:    5.2 - 5 = 0.2s < 0.3s threshold
Output: [Hello world: 0-10]
```

**Example 2: Partial merge**
```
Input:  [Hello: 0-5] [world: 5.1-10] [today: 10.5-15]
Gaps:   0.1s (merge)    0.5s (no merge)
Output: [Hello world: 0-10] [today: 10.5-15]
```

**Example 3: Chain merge**
```
Input:  [First: 0-5] [second: 5.2-10] [third: 10.5-15] [fourth: 15.1-20] [fifth: 20.2-26]
Gaps:   0.2s         0.5s             0.1s             0.2s

Processing right-to-left:
  fifth (20.2) - fourth.end (20) = 0.2 < 0.3 → merge
  [fourth fifth: 15.1-26] - third.end (15) = 0.1 < 0.3 → merge
  [third fourth fifth: 10.5-26] - second.end (10) = 0.5 >= 0.3 → NO merge
  second (5.2) - first.end (5) = 0.2 < 0.3 → merge

Output: [First second: 0-10] [third fourth fifth: 10.5-26]
```

**Example 4: Incomplete segment blocks merge**
```
Input:  [First: 0-5] [second: 5.2-10] [third: 10.5-15] [fourth: 15.1-20] [fifth: 20.2-26] [start: 26.1]
                                                                                           ↑ incomplete

Processing:
  [start: 26.1] has no end/text → skip, don't merge
  Rest merges as before

Output: [First second: 0-10] [third fourth fifth: 10.5-26] [start: 26.1]
```

## Timestamp Tokens (`_get_timestamp_token_text`)

Whisper uses special timestamp tokens rounded to 0.02s precision:
```
<|0.00|>word word<|1.24|><|1.30|>more words<|2.56|>
```

## Dataset Features

```python
{
    "audio": AudioColumnType(),
    "transcript": str,           # with timestamp tokens
    "metadata": {
        "seek": float,           # slice start time
        "duration": float,       # slice length (30s)
        "source": str,           # source_id
        "entry_id": str,         # unique id
        "quality_score": float,  # median word probability
        "source_id": str,
    },
    "has_prev": bool,            # has previous slice context
    "has_timestamps": bool,      # always True
    "prev_transcript": str,      # previous slice transcript
}
```

## Quality Filtering

1. **Per-sample**: Checked via `metadata.quality_score` before processing
2. **Per-segment**: Slices below threshold are abandoned in `_generate_slices()`

## Truncation Handling

When alignment status is `TRUNCATED`:
1. Audio is truncated at `segment_result.truncate_at` timestamp
2. Only aligned segments (up to mismatch) are used
3. Dataset is generated from valid portion only

---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for {{ pretty_name | default("Dataset Name", true) }}

<!-- Provide a quick summary of the dataset. -->

{{ dataset_summary | default("", true) }}

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

{{ dataset_description | default("", true) }}

- **License:** {{ license | default("[More Information Needed]", true) }}

## Dataset Structure

### Data Fields

Each example contains one complete lesson recording with its reference transcript:

- `audio`: Audio column (MP3, 16000 Hz) — full lesson recording
- `transcript`: Reference transcript string (plain text, no timestamp tokens)
- `metadata`: Dictionary with:
  - `id`: Integer lesson ID
  - `name`: Teacher/speaker identifier
  - `daf`: Daf Yomi tractate and page (e.g. `menachot_106`)

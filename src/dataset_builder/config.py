from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secret")


@dataclass
class Config:
    transcription_bucket: str = field(default_factory=lambda: os.getenv("TRANSCRIPTION_BUCKET", "final-transcription"))
    audio_bucket: str = field(default_factory=lambda: os.getenv("AUDIO_BUCKET", "portal-daf-yomi-audio"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    aws_profile: str = field(default_factory=lambda: os.getenv("AWS_PROFILE", "portal"))
    output_dataset_path: str = field(default_factory=lambda: os.getenv("OUTPUT_DATASET_PATH", "output/dataset"))
    card_language: str = field(default_factory=lambda: os.getenv("CARD_LANGUAGE", "he"))
    card_license: str = field(default_factory=lambda: os.getenv("CARD_LICENSE", "cc-by-4.0"))
    card_description: str = field(default_factory=lambda: os.getenv("CARD_DESCRIPTION", ""))
    card_pretty_name: str = field(default_factory=lambda: os.getenv("CARD_PRETTY_NAME", ""))
    card_template_path: str | None = field(default_factory=lambda: os.getenv("CARD_TEMPLATE_PATH"))
    max_shard_size: str = field(default_factory=lambda: os.getenv("MAX_SHARD_SIZE", "500MB"))
    test_size: float = field(default_factory=lambda: float(os.getenv("TEST_SIZE", "0.1")))
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))

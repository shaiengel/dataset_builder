from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secret")


@dataclass
class Config:
    transcription_bucket: str = "final-transcription"
    audio_bucket: str = "portal-daf-yomi-audio"
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    aws_profile: str = field(default_factory=lambda: os.getenv("AWS_PROFILE", "portal"))
    output_dataset_path: str = field(default_factory=lambda: os.getenv("OUTPUT_DATASET_PATH", "output/dataset"))

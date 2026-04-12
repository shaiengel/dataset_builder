from dataclasses import dataclass
import os


@dataclass
class Config:
    transcription_bucket: str = "final-transcription"
    audio_bucket: str = "portal-daf-yomi-audio"
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_profile: str = os.getenv("AWS_PROFILE", "portal")

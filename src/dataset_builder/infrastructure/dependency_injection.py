import boto3
from dependency_injector import containers, providers

from dataset_builder.config import Config
from dataset_builder.infrastructure.json_parser import JsonParser
from dataset_builder.infrastructure.s3_client import S3Client
from dataset_builder.infrastructure.segment_parser import SegmentParser
from dataset_builder.infrastructure.vtt_parser import VttParser
from dataset_builder.infrastructure.whisper_dataset_generator import WhisperDatasetGenerator
from dataset_builder.services.processor import LessonProcessor
from dataset_builder.services.reader import DatasetReader


def _create_session(config: Config):
    return boto3.Session(
        profile_name=config.aws_profile,
        region_name=config.aws_region,
    )


class DependenciesContainer(containers.DeclarativeContainer):
    config = providers.Singleton(Config)

    session = providers.Singleton(_create_session, config=config)

    s3_boto_client = providers.Singleton(
        lambda sess: sess.client("s3"),
        sess=session,
    )

    s3_client = providers.Singleton(S3Client, client=s3_boto_client)

    reader = providers.Singleton(
        DatasetReader,
        s3_client=s3_client,
        config=config,
    )

    json_parser = providers.Singleton(JsonParser)

    vtt_parser = providers.Singleton(VttParser)

    segment_parser = providers.Singleton(SegmentParser)

    dataset_generator = providers.Singleton(WhisperDatasetGenerator)

    processor = providers.Singleton(
        LessonProcessor,
        reader=reader,
        json_parser=json_parser,
        vtt_parser=vtt_parser,
        segment_parser=segment_parser,
        dataset_generator=dataset_generator,
    )

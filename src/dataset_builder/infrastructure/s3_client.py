import logging

from botocore.exceptions import ClientError

from dataset_builder.domain.file_manager import FileManager

logger = logging.getLogger(__name__)


class S3Client(FileManager):
    def __init__(self, client) -> None:
        self._client = client

    def get_text(self, bucket: str, key: str) -> str | None:
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read().decode("utf-8")
        except ClientError as e:
            logger.error(f"Failed to get s3://{bucket}/{key}: {e}")
            return None

    def get_bytes(self, bucket: str, key: str) -> bytes | None:
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Failed to get s3://{bucket}/{key}: {e}")
            return None

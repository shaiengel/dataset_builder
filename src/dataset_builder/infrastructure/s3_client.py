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

    def list_keys(self, bucket: str, suffix: str = "") -> list[str]:
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not suffix or key.endswith(suffix):
                    keys.append(key)
        logger.info(f"Listed {len(keys)} keys in s3://{bucket} (suffix={suffix!r})")
        return keys

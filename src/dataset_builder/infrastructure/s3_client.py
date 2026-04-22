import logging
import time

from botocore.exceptions import ClientError, EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError

from dataset_builder.domain.file_manager import FileManager

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError, OSError)
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 5  # seconds


def _with_retry(fn, label: str):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except _TRANSIENT_ERRORS as e:
            if attempt == _MAX_RETRIES:
                logger.error("Giving up on %s after %d attempts: %s", label, _MAX_RETRIES, e)
                return None
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("Network error on %s (attempt %d/%d), retrying in %ds: %s", label, attempt, _MAX_RETRIES, delay, e)
            time.sleep(delay)


class S3Client(FileManager):
    def __init__(self, client) -> None:
        self._client = client

    def get_text(self, bucket: str, key: str) -> str | None:
        label = f"s3://{bucket}/{key}"

        def _read_text():
            obj = self._client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode("utf-8")

        try:
            return _with_retry(_read_text, label)
        except ClientError as e:
            logger.error(f"Failed to get {label}: {e}")
            return None

    def get_bytes(self, bucket: str, key: str) -> bytes | None:
        label = f"s3://{bucket}/{key}"

        def _read_bytes():
            obj = self._client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()

        try:
            return _with_retry(_read_bytes, label)
        except ClientError as e:
            logger.error(f"Failed to get {label}: {e}")
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

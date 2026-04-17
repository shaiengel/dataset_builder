from abc import ABC, abstractmethod


class FileManager(ABC):
    @abstractmethod
    def get_text(self, bucket: str, key: str) -> str | None:
        """Read a file and return its contents as UTF-8 text."""
        ...

    @abstractmethod
    def get_bytes(self, bucket: str, key: str) -> bytes | None:
        """Read a file and return its raw bytes."""
        ...

    @abstractmethod
    def list_keys(self, bucket: str, suffix: str = "") -> list[str]:
        """List all keys in a bucket, optionally filtered by suffix."""
        ...

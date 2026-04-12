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

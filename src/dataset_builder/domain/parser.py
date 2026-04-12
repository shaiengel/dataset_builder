from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Parser(ABC, Generic[T]):
    @abstractmethod
    def parse(self, content: str) -> T:
        """Parse string content into domain model."""
        ...

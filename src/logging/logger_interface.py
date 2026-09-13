from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    def __init__(self, buffer_size: int = 5):
        self.buffer = []
        self.buffer_size = buffer_size

    def _buffer_is_full(self) -> bool:
        return len(self.buffer) == self.buffer_size

    def _print_buffer(self) -> None:
        for row_lines in zip(*self.buffer):  # type: ignore
            print("".join(line.ljust(34) for line in row_lines))
        print()
        self.buffer.clear()

    @abstractmethod
    def log(self, *args: Any, **kwargs: Any) -> None:
        pass

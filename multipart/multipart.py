from __future__ import annotations

from typing import Any, Dict, Tuple


def parse_options_header(value: Any) -> Tuple[Any, Dict[bytes, bytes]]:
    """Minimal stub used for tests."""
    return value, {}


class MultipartParser:
    """Non-functional stub used to satisfy FastAPI's import checks."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def write(self, data: bytes) -> None:  # pragma: no cover - no-op
        self._last_write = data

    def finalize(self) -> None:  # pragma: no cover - no-op
        pass

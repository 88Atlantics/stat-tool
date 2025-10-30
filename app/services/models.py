from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class ImagePayload:
    """Container representing an encoded image returned by a tool."""

    content_type: str
    encoding: str
    data: str

    def asdict(self) -> Dict[str, str]:
        return {
            "content_type": self.content_type,
            "encoding": self.encoding,
            "data": self.data,
        }


@dataclass
class ToolResult:
    name: str
    summary: str
    images: List[ImagePayload]


@dataclass
class AgentResult:
    summary: str
    tool_summaries: Dict[str, str]
    tool_images: Dict[str, List[ImagePayload]]


@dataclass
class QueryPlan:
    """Structured interpretation of a natural language analysis request."""

    tools: Sequence[str]
    tickers: Sequence[str]
    start_date: dt.date | None
    end_date: dt.date | None

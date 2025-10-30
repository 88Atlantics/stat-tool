from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class ToolResult:
    name: str
    summary: str
    images: List[str]


@dataclass
class AgentResult:
    summary: str
    tool_summaries: Dict[str, str]
    tool_images: Dict[str, List[str]]


@dataclass
class QueryPlan:
    """Structured interpretation of a natural language analysis request."""

    tools: Sequence[str]
    tickers: Sequence[str]
    start_date: dt.date | None
    end_date: dt.date | None

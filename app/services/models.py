from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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

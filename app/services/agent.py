from __future__ import annotations

from typing import Callable, Iterable

from app.services.models import AgentResult, ToolResult
from app.services.preprocess import PriceMatrix
from app.tools import rsi, sma, zscore


ToolFn = Callable[[PriceMatrix], ToolResult]


class AgentService:
    """Simple rule-based agent that dispatches statistical tools."""

    def __init__(self) -> None:
        self._tool_map: dict[str, ToolFn] = {
            "zscore": zscore.analyze_zscore,
            "rsi": rsi.analyze_rsi,
            "sma": sma.analyze_sma,
        }

    def select_tools(self, query: str) -> Iterable[tuple[str, ToolFn]]:
        lowered = query.lower()
        selected: dict[str, ToolFn] = {}
        for keyword, tool in self._tool_map.items():
            if keyword in lowered:
                selected[keyword] = tool
        if not selected:
            selected = self._tool_map
        return selected.items()

    def run_analysis(self, query: str, data: PriceMatrix) -> AgentResult:
        tool_summaries: dict[str, str] = {}
        tool_images: dict[str, list[str]] = {}
        aggregate_notes: list[str] = []

        for _, tool in self.select_tools(query):
            result = tool(data)
            tool_summaries[result.name] = result.summary
            tool_images[result.name] = result.images
            aggregate_notes.append(f"{result.name.upper()}: {result.summary}")

        summary = " \n".join(aggregate_notes) if aggregate_notes else "No insights generated."
        return AgentResult(summary=summary, tool_summaries=tool_summaries, tool_images=tool_images)

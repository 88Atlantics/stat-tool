from __future__ import annotations

import json
import os
import textwrap
from typing import Callable, Iterable, Sequence

try:  # pragma: no cover - network interactions excluded from tests
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from app.services.models import AgentResult, ToolResult
from app.services.preprocess import PriceMatrix
from app.tools import rsi, sma, zscore


ToolFn = Callable[[PriceMatrix], ToolResult]


class AgentService:
    """Simple rule-based agent that dispatches statistical tools."""

    def __init__(self, llm_client: object | None = None) -> None:
        self._tool_map: dict[str, ToolFn] = {
            "zscore": zscore.analyze_zscore,
            "rsi": rsi.analyze_rsi,
            "sma": sma.analyze_sma,
        }
        self._llm_client = llm_client or self._build_llm_client()

    def _build_llm_client(self) -> object | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None
        try:  # pragma: no cover - network client initialisation
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _keyword_select(self, query: str) -> Iterable[tuple[str, ToolFn]]:
        lowered = query.lower()
        selected: dict[str, ToolFn] = {}
        for keyword, tool in self._tool_map.items():
            if keyword in lowered:
                selected[keyword] = tool
        if not selected:
            selected = self._tool_map
        return selected.items()

    def _parse_tool_selection(self, text: str) -> Sequence[str]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict) and "tools" in payload:
            items = payload["tools"]
        elif isinstance(payload, list):
            items = payload
        else:
            return []
        names: list[str] = []
        for item in items:
            if isinstance(item, str) and item in self._tool_map:
                names.append(item)
            elif isinstance(item, dict):
                candidate = item.get("tool") if isinstance(item, dict) else None
                if isinstance(candidate, str) and candidate in self._tool_map:
                    names.append(candidate)
        return names

    def interpret_query(self, query: str) -> Iterable[tuple[str, ToolFn]]:
        if not self._llm_client:
            return self._keyword_select(query)

        prompt = textwrap.dedent(
            f"""
            Available analytics tools: {', '.join(self._tool_map.keys())}.
            Return a strict JSON object using the schema {{"tools": [tool_names...]}}.
            Choose the tools that best answer the question. Only use tool names from the available list.
            User query: "{query}".
            """
        ).strip()

        try:  # pragma: no cover - exercised via network
            response = self._llm_client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": "You select quantitative finance tools."},
                    {"role": "user", "content": prompt},
                ],
            )
            text_parts: list[str] = []
            for item in getattr(response, "output", []):
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        text_parts.append(getattr(content, "text", ""))
            if not text_parts and hasattr(response, "output_text"):
                text_parts.append(getattr(response, "output_text"))
            parsed_names = self._parse_tool_selection("".join(text_parts))
            if parsed_names:
                return [(name, self._tool_map[name]) for name in dict.fromkeys(parsed_names)]
        except Exception:
            pass

        return self._keyword_select(query)

    def _generate_summary_with_llm(self, query: str, notes: dict[str, str]) -> str | None:
        if not self._llm_client:
            return None
        context = json.dumps(notes)
        prompt = textwrap.dedent(
            f"""
            You are an investment analyst. Summarise the following tool outputs in a concise paragraph.
            User query: {query}
            Tool outputs (JSON): {context}
            Provide actionable insights and mention notable divergences.
            """
        ).strip()
        try:  # pragma: no cover - network call
            response = self._llm_client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": "You craft succinct investment commentary."},
                    {"role": "user", "content": prompt},
                ],
            )
            text_parts: list[str] = []
            for item in getattr(response, "output", []):
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        text_parts.append(getattr(content, "text", ""))
            if not text_parts and hasattr(response, "output_text"):
                text_parts.append(getattr(response, "output_text"))
            combined = "".join(text_parts).strip()
            return combined or None
        except Exception:
            return None

    def run_analysis(self, query: str, data: PriceMatrix) -> AgentResult:
        tool_summaries: dict[str, str] = {}
        tool_images: dict[str, list[str]] = {}
        aggregate_notes: dict[str, str] = {}

        for name, tool in self.interpret_query(query):
            result = tool(data)
            tool_summaries[result.name] = result.summary
            tool_images[result.name] = result.images
            aggregate_notes[result.name] = result.summary

        summary = self._generate_summary_with_llm(query, aggregate_notes)
        if not summary:
            summary = " ".join(
                f"{name.upper()}: {note}" for name, note in aggregate_notes.items()
            ) or "No insights generated. (LLM unavailable)"
        return AgentResult(summary=summary, tool_summaries=tool_summaries, tool_images=tool_images)

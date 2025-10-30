from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import re
import textwrap
from typing import Callable, Sequence

try:  # pragma: no cover - network interactions excluded from tests
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from app.services.models import AgentResult, QueryPlan, ToolResult
from app.services.preprocess import PriceMatrix
from app.tools import rsi, sma, zscore


ToolFn = Callable[[PriceMatrix], ToolResult]

_ENGLISH_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_CHINESE_DIGIT_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


class AgentService:
    """Simple rule-based agent that dispatches statistical tools."""

    def __init__(self, llm_client: object | None = None) -> None:
        self._tool_map: dict[str, ToolFn] = {
            "zscore": zscore.analyze_zscore,
            "rsi": rsi.analyze_rsi,
            "sma": sma.analyze_sma,
        }
        self._llm_client = llm_client or self._build_llm_client()
        self._keyword_ticker_map: dict[str, str] = {
            "apple": "AAPL",
            "苹果": "AAPL",
            "tesla": "TSLA",
            "特斯拉": "TSLA",
            "microsoft": "MSFT",
            "微软": "MSFT",
        }

    def _build_llm_client(self) -> object | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None
        try:  # pragma: no cover - network client initialisation
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _today(self) -> dt.date:
        return dt.date.today()

    def current_date(self) -> dt.date:
        """Expose today's date for coordination with other components."""
        return self._today()

    def _months_ago(self, reference: dt.date, months: int) -> dt.date:
        year = reference.year
        month = reference.month - months
        while month <= 0:
            month += 12
            year -= 1
        day = min(reference.day, calendar.monthrange(year, month)[1])
        return dt.date(year, month, day)

    def default_lookback(self, months: int = 12, end_date: dt.date | None = None) -> dt.date:
        reference = end_date or self.current_date()
        return self._months_ago(reference, months)

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

    def _parse_date(self, value: object) -> dt.date | None:
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            try:
                return dt.date.fromisoformat(value.strip())
            except ValueError:
                return None
        return None

    def _ensure_sequence(self, value: object) -> Sequence[object]:
        if isinstance(value, (list, tuple)):
            return value
        if value is None:
            return []
        return [value]

    def _parse_chinese_numeral(self, token: str) -> int | None:
        if not token:
            return None
        if token == "十":
            return 10
        if token.endswith("十") and len(token) == 2:
            leading = _CHINESE_DIGIT_MAP.get(token[0])
            if leading is None:
                return None
            return leading * 10
        if "十" in token:
            prefix, suffix = token.split("十", 1)
            prefix_value = 1 if prefix == "" else _CHINESE_DIGIT_MAP.get(prefix)
            suffix_value = 0 if suffix == "" else _CHINESE_DIGIT_MAP.get(suffix)
            if prefix_value is None or suffix_value is None:
                return None
            return prefix_value * 10 + suffix_value
        value = 0
        for char in token:
            digit = _CHINESE_DIGIT_MAP.get(char)
            if digit is None:
                return None
            value = value * 10 + digit
        return value

    def _parse_numeric_token(self, token: str) -> int | None:
        cleaned = token.strip().lower()
        if not cleaned:
            return None
        if cleaned.isdigit():
            return int(cleaned)
        if cleaned in _ENGLISH_NUMBER_MAP:
            return _ENGLISH_NUMBER_MAP[cleaned]
        if cleaned in {"half", "half-year", "half year"}:
            return 6
        chinese_value = self._parse_chinese_numeral(token.strip())
        if chinese_value is not None:
            return chinese_value
        return None

    def _subtract_period(self, end_date: dt.date, quantity: int, unit: str) -> dt.date | None:
        if quantity <= 0:
            return None
        normalized = unit.lower()
        if normalized.startswith("day"):
            return end_date - dt.timedelta(days=quantity)
        if normalized.startswith("week"):
            return end_date - dt.timedelta(weeks=quantity)
        if normalized.startswith("month"):
            return self._months_ago(end_date, quantity)
        if normalized.startswith("year"):
            return self._months_ago(end_date, quantity * 12)
        if normalized in {"m", "mo", "mos"}:
            return self._months_ago(end_date, quantity)
        if normalized in {"y", "yr", "yrs"}:
            return self._months_ago(end_date, quantity * 12)
        if normalized in {"w", "wk", "wks"}:
            return end_date - dt.timedelta(weeks=quantity)
        if normalized in {"d"}:
            return end_date - dt.timedelta(days=quantity)
        return None

    def _interpret_relative_window(self, value: object, default_end: dt.date | None = None) -> tuple[dt.date | None, dt.date | None]:
        end_date = default_end or self._today()
        quantity: int | None = None
        unit: str | None = None
        if isinstance(value, dict):
            quantity = self._parse_numeric_token(str(value.get("quantity", "")))
            unit = str(value.get("unit", "")).lower()
        elif isinstance(value, str):
            match = re.search(
                r"(?P<quantity>\d+|[A-Za-z\u4e00-\u9fa5两半]+)\s*(?P<unit>years?|yrs?|y|months?|mos?|m|weeks?|wks?|w|days?|d)",
                value.lower(),
            )
            if match:
                quantity = self._parse_numeric_token(match.group("quantity"))
                unit = match.group("unit")
        if quantity and unit:
            start_date = self._subtract_period(end_date, quantity, unit)
            if start_date:
                return start_date, end_date
        return None, default_end

    def _extract_relative_period(self, query: str) -> tuple[dt.date | None, dt.date | None]:
        lowered = query.lower()
        today = self._today()
        english_match = re.search(
            r"(?:past|last|trailing)\s+(?P<quantity>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?P<unit>day|days|week|weeks|month|months|year|years)",
            lowered,
        )
        if english_match:
            quantity = self._parse_numeric_token(english_match.group("quantity"))
            unit = english_match.group("unit")
            if quantity and unit:
                start = self._subtract_period(today, quantity, unit)
                if start:
                    return start, today

        simple_match = re.search(r"(?P<quantity>\d+)(?P<unit>y|yr|yrs|m|mo|mos|w|wk|wks|d)", lowered)
        if simple_match:
            quantity = self._parse_numeric_token(simple_match.group("quantity"))
            unit = simple_match.group("unit")
            if quantity and unit:
                start = self._subtract_period(today, quantity, unit)
                if start:
                    return start, today

        chinese_match = re.search(r"过去(?P<quantity>[\u4e00-\u9fa5两半\d]+)(?P<unit>年|个月|月|周|星期|天|日)", query)
        if chinese_match:
            quantity = self._parse_numeric_token(chinese_match.group("quantity"))
            unit = chinese_match.group("unit")
            unit_map = {
                "年": "year",
                "个月": "month",
                "月": "month",
                "周": "week",
                "星期": "week",
                "天": "day",
                "日": "day",
            }
            mapped_unit = unit_map.get(unit, unit)
            if quantity and mapped_unit:
                start = self._subtract_period(today, quantity, mapped_unit)
                if start:
                    return start, today

        if "过去半年" in query:
            start = self._subtract_period(today, 6, "month")
            if start:
                return start, today

        if "过去一年" in query or "last year" in lowered or "past year" in lowered:
            start = self._subtract_period(today, 1, "year")
            if start:
                return start, today

        return None, None

    def _extract_tickers(self, query: str) -> list[str]:
        uppercase_words = re.findall(r"\b([A-Z]{1,5})\b", query)
        tickers: list[str] = []
        for word in uppercase_words:
            if word.lower() in self._tool_map:
                continue
            tickers.append(word.upper())
        lowered = query.lower()
        for keyword, symbol in self._keyword_ticker_map.items():
            if keyword in lowered:
                tickers.append(symbol)
        return list(dict.fromkeys(tickers))

    def _keyword_plan(self, query: str) -> QueryPlan:
        lowered = query.lower()
        tools: list[str] = []
        for name in self._tool_map:
            if name in lowered:
                tools.append(name)
        if not tools:
            if "moving average" in lowered or "sma" in lowered:
                tools.append("sma")
            if "relative strength" in lowered or "rsi" in lowered:
                tools.append("rsi")
            if "z-score" in lowered or "z score" in lowered or "zscore" in lowered:
                tools.append("zscore")
        if not tools:
            tools = list(self._tool_map.keys())
        start_date, end_date = self._extract_relative_period(query)
        tickers = self._extract_tickers(query)
        return QueryPlan(tools=tuple(dict.fromkeys(tools)), tickers=tuple(tickers), start_date=start_date, end_date=end_date)

    def _resolve_tools(self, names: Sequence[str] | None) -> list[tuple[str, ToolFn]]:
        if not names:
            return list(self._tool_map.items())
        resolved: list[tuple[str, ToolFn]] = []
        for name in names:
            tool = self._tool_map.get(name)
            if tool:
                resolved.append((name, tool))
        return resolved or list(self._tool_map.items())

    def _interpret_with_llm(self, query: str) -> QueryPlan | None:
        if not self._llm_client:
            return None

        prompt = textwrap.dedent(
            f"""
            You act as a planner for an investment statistics agent.
            Available analytics tools: {', '.join(self._tool_map.keys())}.
            Return a strict JSON object using the schema:
            {{
                "tools": [tool_names...],
                "tickers": [ticker_symbols...],
                "start_date": "YYYY-MM-DD" | null,
                "end_date": "YYYY-MM-DD" | null,
                "lookback": {{"quantity": <int>, "unit": "day/week/month/year"}} | null
            }}
            - Tool names must come from the available list only.
            - Ticker symbols must be upper-case.
            - If the user specifies a relative timeframe (e.g. last six months), express it via the lookback object.
            - Omit fields that are not specified by the user by setting them to null.
            User query: "{query}".
            """
        ).strip()

        try:  # pragma: no cover - exercised via network
            response = self._llm_client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": "You extract structured plans for financial data analysis."},
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
            raw_text = "".join(text_parts).strip()
            payload = json.loads(raw_text)
        except Exception:
            return None

        tools = [
            name
            for name in (str(item).lower() for item in self._ensure_sequence(payload.get("tools")))
            if name in self._tool_map
        ]
        tickers = [
            str(item).upper()
            for item in self._ensure_sequence(payload.get("tickers"))
            if isinstance(item, (str, bytes)) and str(item).strip()
        ]
        start_date = self._parse_date(payload.get("start_date"))
        end_date = self._parse_date(payload.get("end_date"))
        if payload.get("lookback") and not start_date:
            start_date, inferred_end = self._interpret_relative_window(payload["lookback"], end_date)
            if inferred_end and not end_date:
                end_date = inferred_end

        return QueryPlan(
            tools=tuple(dict.fromkeys(tools)),
            tickers=tuple(dict.fromkeys(tickers)),
            start_date=start_date,
            end_date=end_date,
        )

    def interpret_query(self, query: str) -> QueryPlan:
        heuristic_plan = self._keyword_plan(query)
        llm_plan = self._interpret_with_llm(query)
        if not llm_plan:
            return heuristic_plan

        tools = tuple(llm_plan.tools) if llm_plan.tools else heuristic_plan.tools
        tickers = tuple(llm_plan.tickers) if llm_plan.tickers else heuristic_plan.tickers
        start_date = llm_plan.start_date or heuristic_plan.start_date
        end_date = llm_plan.end_date or heuristic_plan.end_date
        return QueryPlan(tools=tools, tickers=tickers, start_date=start_date, end_date=end_date)

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

    def run_analysis(
        self, query: str, data: PriceMatrix, plan: QueryPlan | None = None
    ) -> AgentResult:
        tool_summaries: dict[str, str] = {}
        tool_images: dict[str, list[str]] = {}
        aggregate_notes: dict[str, str] = {}

        tool_selection = self._resolve_tools(plan.tools if plan else None)
        for name, tool in tool_selection:
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

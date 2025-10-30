from __future__ import annotations

import csv
import datetime as dt
import io
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class PriceMatrix:
    dates: List[dt.date]
    series: Dict[str, List[float]]

    @property
    def tickers(self) -> List[str]:
        return list(self.series.keys())

    def is_empty(self) -> bool:
        return not self.dates or not self.series


RawRecord = Dict[str, object]


def clean_stock_data(records: Iterable[RawRecord]) -> PriceMatrix:
    normalized: List[tuple[dt.date, str, float]] = []
    for row in records:
        try:
            date_value = row.get("Date") or row.get("date")
            ticker_value = row.get("Ticker") or row.get("ticker")
            close_value = row.get("Close") or row.get("close")
            if date_value is None or ticker_value is None or close_value is None:
                continue
            if isinstance(date_value, dt.datetime):
                date = date_value.date()
            elif isinstance(date_value, dt.date):
                date = date_value
            else:
                date = dt.date.fromisoformat(str(date_value))
            ticker = str(ticker_value).upper()
            price = float(close_value)
            normalized.append((date, ticker, price))
        except Exception:
            continue

    if not normalized:
        return PriceMatrix(dates=[], series={})

    normalized.sort(key=lambda item: (item[0], item[1]))
    tickers = sorted({item[1] for item in normalized})
    date_map: Dict[dt.date, Dict[str, float]] = {}
    for date, ticker, price in normalized:
        date_map.setdefault(date, {})[ticker] = price

    sorted_dates = sorted(date_map.keys())
    filled_series: Dict[str, List[float]] = {ticker: [] for ticker in tickers}
    latest_values: Dict[str, float | None] = {ticker: None for ticker in tickers}

    for current_date in sorted_dates:
        for ticker in tickers:
            if ticker in date_map[current_date]:
                latest_values[ticker] = date_map[current_date][ticker]
            filled_series[ticker].append(latest_values[ticker])

    valid_indices: List[int] = []
    for idx in range(len(sorted_dates)):
        if all(filled_series[ticker][idx] is not None for ticker in tickers):
            valid_indices.append(idx)

    filtered_dates = [sorted_dates[idx] for idx in valid_indices]
    filtered_series = {
        ticker: [filled_series[ticker][idx] for idx in valid_indices] for ticker in tickers
    }

    return PriceMatrix(dates=filtered_dates, series=filtered_series)


def _derive_ticker_from_filename(filename: str) -> str | None:
    stem = filename.rsplit(".", 1)[0]
    candidate = re.sub(r"[^A-Za-z0-9]+", "", stem)
    return candidate.upper() if candidate else None


def _parse_matrix_layout(
    rows: List[List[str]], fallback_symbol: str | None
) -> List[RawRecord]:
    if not rows:
        return []

    # Normalise rows by trimming whitespace and removing empties
    normalized_rows: List[List[str]] = []
    for row in rows:
        cleaned = [cell.strip() for cell in row]
        if any(cell for cell in cleaned):
            normalized_rows.append(cleaned)

    if not normalized_rows:
        return []

    header_row = normalized_rows[0]
    ticker_row = normalized_rows[1] if len(normalized_rows) > 1 else []

    date_row_index = None
    for idx, row in enumerate(normalized_rows):
        if row and row[0].lower() == "date":
            date_row_index = idx
            break

    if date_row_index is None or date_row_index + 1 >= len(normalized_rows):
        return []

    column_labels = header_row[1:] if len(header_row) > 1 else []
    close_index = None
    for idx, label in enumerate(column_labels, start=1):
        lowered = label.lower()
        if lowered in {"close", "adj close", "price", "close price", "last"}:
            close_index = idx
            break

    if close_index is None and column_labels:
        close_index = 1

    derived_ticker = fallback_symbol
    if not derived_ticker:
        if close_index is not None and len(ticker_row) > close_index:
            candidate = ticker_row[close_index]
            if candidate:
                derived_ticker = candidate.upper()
        if not derived_ticker:
            for cell in ticker_row[1:]:
                if cell:
                    derived_ticker = cell.upper()
                    break

    if not derived_ticker:
        return []

    data_rows = normalized_rows[date_row_index + 1 :]
    parsed: List[RawRecord] = []
    for row in data_rows:
        if not row or not row[0]:
            continue
        date_value = row[0]
        close_value = None
        if close_index is not None and len(row) > close_index and row[close_index]:
            close_value = row[close_index]
        else:
            for cell in row[1:]:
                if cell:
                    close_value = cell
                    break
        if not close_value:
            continue
        parsed.append({"Date": date_value, "Ticker": derived_ticker, "Close": close_value})

    return parsed


def parse_uploaded_prices(
    content: bytes,
    filename: str | None = None,
    fallback_tickers: Sequence[str] | None = None,
) -> List[RawRecord]:
    if not content:
        return []

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="ignore")

    dialect = csv.excel
    if text.strip():
        sample = "\n".join(text.splitlines()[:2])
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            pass
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    fallback_symbols = [
        str(symbol).strip().upper()
        for symbol in (fallback_tickers or [])
        if isinstance(symbol, (str, bytes)) and str(symbol).strip()
    ]
    fallback_symbol = fallback_symbols[0] if fallback_symbols else None
    if not fallback_symbol and filename:
        fallback_symbol = _derive_ticker_from_filename(filename)
    records: List[RawRecord] = []
    for row in reader:
        date_value = row.get("Date") or row.get("date")
        ticker_value = (
            row.get("Ticker")
            or row.get("ticker")
            or row.get("Symbol")
            or row.get("symbol")
            or fallback_symbol
        )
        close_value = row.get("Close") or row.get("close") or row.get("Adj Close")
        if not date_value or not ticker_value or close_value in (None, ""):
            continue
        records.append(
            {
                "Date": date_value,
                "Ticker": str(ticker_value).strip().upper(),
                "Close": close_value,
            }
        )

    if records:
        return records

    structured_rows = list(csv.reader(io.StringIO(text)))
    matrix_records = _parse_matrix_layout(structured_rows, fallback_symbol)
    if matrix_records:
        return matrix_records

    if filename and filename.lower().endswith(".json"):
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        date_value = item.get("Date") or item.get("date")
                        ticker_value = (
                            item.get("Ticker")
                            or item.get("ticker")
                            or item.get("Symbol")
                            or item.get("symbol")
                            or fallback_symbol
                        )
                        close_value = item.get("Close") or item.get("close")
                        if date_value and ticker_value and close_value is not None:
                            records.append(
                                {
                                    "Date": date_value,
                                    "Ticker": str(ticker_value).strip().upper(),
                                    "Close": close_value,
                                }
                            )
        except Exception:
            return []

    return records

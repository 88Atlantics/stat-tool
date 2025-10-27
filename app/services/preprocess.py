from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List


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

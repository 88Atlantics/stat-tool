from __future__ import annotations

import datetime as dt
from typing import Iterable, List

try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yf = None


def load_market_data(
    tickers: Iterable[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> List[dict[str, object]]:
    symbols = [ticker.upper() for ticker in tickers]
    if not symbols or yf is None:
        return []

    data = yf.download(
        tickers=symbols,
        start=start.isoformat() if start else None,
        end=end.isoformat() if end else None,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    records: List[dict[str, object]] = []
    if hasattr(data, "columns") and getattr(data, "columns", None) is not None:
        columns = getattr(data, "columns")
        if getattr(columns, "nlevels", 1) == 2:
            for ticker in symbols:
                try:
                    close_series = data[("Close", ticker)]
                except KeyError:
                    continue
                for date, value in close_series.dropna().items():
                    records.append({"Date": date.date(), "Ticker": ticker, "Close": float(value)})
            return records
        if "Close" in columns:
            close_series = data["Close"].dropna()
            ticker = symbols[0]
            for date, value in close_series.items():
                records.append({"Date": date.date(), "Ticker": ticker, "Close": float(value)})
            return records
    return records

import asyncio
import datetime as dt
import io
import pathlib
import sys

import pytest
from fastapi import HTTPException, UploadFile

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.routers import analysis
from app.routers.analysis import agent_service


def _sample_csv_payload() -> str:
    base_date = dt.date(2024, 1, 1)
    rows = ["Date,Ticker,Close"]
    for i in range(30):
        rows.append(
            f"{(base_date + dt.timedelta(days=i)).isoformat()},AAA,{100 + i}"
        )
        rows.append(
            f"{(base_date + dt.timedelta(days=i)).isoformat()},BBB,{120 - i}"
        )
    return "\n".join(rows)


def _csv_without_ticker() -> str:
    base_date = dt.date(2024, 2, 1)
    rows = ["Date,Close"]
    for i in range(10):
        rows.append(f"{(base_date + dt.timedelta(days=i)).isoformat()},{150 + i}")
    return "\n".join(rows)


def _market_records(ticker: str, start: dt.date, days: int = 60) -> list[dict[str, object]]:
    return [
        {"Date": start + dt.timedelta(days=i), "Ticker": ticker, "Close": 200 + i}
        for i in range(days)
    ]


def _invoke_run_analysis(
    query: str,
    *,
    csv_payload: str | None = None,
    tickers: str | None = None,
    start: str | None = None,
    end: str | None = None,
):
    upload: UploadFile | None = None
    if csv_payload is not None:
        upload = UploadFile(
            filename="upload.csv",
            file=io.BytesIO(csv_payload.encode("utf-8")),
        )
    return asyncio.run(
        analysis.run_analysis(
            query=query,
            tickers=tickers,
            start_date=start,
            end_date=end,
            upload_file=upload,
        )
    )


def test_analysis_endpoint_returns_results():
    csv_payload = _sample_csv_payload()
    result = _invoke_run_analysis(
        "Please run zscore, rsi and sma analysis",
        csv_payload=csv_payload,
    )
    assert set(result.tool_summaries.keys()) == {"zscore", "rsi", "sma"}
    for images in result.images.values():
        assert all(isinstance(img, str) and len(img) > 10 for img in images)


def test_missing_data_returns_error():
    with pytest.raises(HTTPException) as excinfo:
        _invoke_run_analysis("test")
    assert excinfo.value.status_code == 400


def test_uploaded_file_without_ticker_is_used():
    csv_payload = _csv_without_ticker()
    result = _invoke_run_analysis("苹果过去六个月的sma", csv_payload=csv_payload)
    assert "sma" in result.tool_summaries
    assert result.images["sma"]


def test_query_without_ticker_loads_market_data(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_market_data(tickers, start, end):  # type: ignore[override]
        captured["tickers"] = list(tickers)
        captured["start"] = start
        captured["end"] = end
        return _market_records(list(tickers)[0], start or dt.date(2024, 1, 1))

    monkeypatch.setattr("app.routers.analysis.load_market_data", fake_load_market_data)
    monkeypatch.setattr(agent_service, "_today", lambda: dt.date(2024, 7, 1))
    monkeypatch.setattr(agent_service, "current_date", lambda: dt.date(2024, 7, 1))

    result = _invoke_run_analysis("AAPL past six months sma")
    assert captured["tickers"] == ["AAPL"]
    assert captured["start"] == dt.date(2024, 1, 1)
    assert captured["end"] == dt.date(2024, 7, 1)
    assert "sma" in result.tool_summaries

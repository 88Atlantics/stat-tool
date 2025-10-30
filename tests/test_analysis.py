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


def _csv_price_only_headers() -> str:
    base_date = dt.date(2024, 3, 1)
    rows = ["date,close,high,low,open,volume"]
    for i in range(5):
        rows.append(
            ",".join(
                [
                    (base_date + dt.timedelta(days=i)).isoformat(),
                    str(200 + i),
                    str(205 + i),
                    str(195 + i),
                    str(198 + i),
                    str(1000 + i),
                ]
            )
        )
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
    upload_bytes: bytes | None = None,
    tickers: str | None = None,
    start: str | None = None,
    end: str | None = None,
    filename: str = "upload.csv",
):
    upload: UploadFile | None = None
    payload_bytes: bytes | None = None
    if csv_payload is not None and upload_bytes is not None:
        raise ValueError("Provide either csv_payload or upload_bytes, not both")
    if csv_payload is not None:
        payload_bytes = csv_payload.encode("utf-8")
    elif upload_bytes is not None:
        payload_bytes = upload_bytes

    if payload_bytes is not None:
        upload = UploadFile(filename=filename, file=io.BytesIO(payload_bytes))
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
        assert all(isinstance(img, str) for img in images)
        assert all(
            img.startswith("/static/visuals/") or img.startswith("http")
            for img in images
        )


def test_missing_data_returns_error():
    with pytest.raises(HTTPException) as excinfo:
        _invoke_run_analysis("test")
    assert excinfo.value.status_code == 400


def test_uploaded_file_without_ticker_is_used():
    csv_payload = _csv_without_ticker()
    result = _invoke_run_analysis("Apple last six months sma", csv_payload=csv_payload)
    assert "sma" in result.tool_summaries
    assert result.images["sma"]


def test_price_only_headers_are_accepted():
    csv_payload = _csv_price_only_headers()
    result = _invoke_run_analysis("Please compute sma", csv_payload=csv_payload)
    assert "sma" in result.tool_summaries
    assert all(
        image.startswith("/static/visuals/") or image.startswith("http")
        for image in result.images["sma"]
    )


def test_real_aapl_csv_is_parsed():
    csv_path = pathlib.Path(__file__).with_name("AAPL.csv")
    payload = csv_path.read_bytes()
    result = _invoke_run_analysis(
        "AAPL past six months sma",
        upload_bytes=payload,
        tickers="AAPL",
        start="2020-01-02",
        end="2020-06-30",
        filename="AAPL.csv",
    )
    assert "sma" in result.tool_summaries
    assert all(
        image.startswith("/static/visuals/") or image.startswith("http")
        for image in result.images["sma"]
    )


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
    assert all(
        image.startswith("/static/visuals/") or image.startswith("http")
        for image in result.images["sma"]
    )


def test_static_base_url_prefix(monkeypatch):
    monkeypatch.setenv("PUBLIC_STATIC_BASE_URL", "https://cdn.example.com")
    try:
        result = _invoke_run_analysis("Please compute sma", csv_payload=_csv_without_ticker())
        assert result.images["sma"]
        assert all(
            image.startswith("https://cdn.example.com/static/visuals/")
            for image in result.images["sma"]
        )
    finally:
        monkeypatch.delenv("PUBLIC_STATIC_BASE_URL", raising=False)


def test_local_static_path_used_when_base_url_absent(monkeypatch):
    monkeypatch.delenv("PUBLIC_STATIC_BASE_URL", raising=False)
    result = _invoke_run_analysis("Please compute sma", csv_payload=_csv_without_ticker())
    assert result.images["sma"]
    assert all(
        image.startswith("/static/visuals/")
        for image in result.images["sma"]
    )

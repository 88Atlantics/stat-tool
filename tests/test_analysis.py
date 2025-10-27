import datetime as dt
import pathlib
import sys

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app

client = TestClient(app)


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


def test_analysis_endpoint_returns_results():
    csv_payload = _sample_csv_payload()
    response = client.post(
        "/analysis/",
        data={"query": "Please run zscore, rsi and sma analysis"},
        files={"upload_file": ("prices.csv", csv_payload, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "analysis" in body
    assert set(body["tool_summaries"].keys()) == {"zscore", "rsi", "sma"}
    for images in body["images"].values():
        assert all(isinstance(img, str) and len(img) > 10 for img in images)


def test_missing_data_returns_error():
    response = client.post("/analysis/", data={"query": "test"})
    assert response.status_code == 400

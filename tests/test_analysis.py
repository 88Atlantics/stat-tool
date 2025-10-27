import datetime as dt
import pathlib
import sys

from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app

client = TestClient(app)


def _sample_upload_payload():
    base_date = dt.date(2024, 1, 1)
    records = []
    for i in range(30):
        records.append({"date": (base_date + dt.timedelta(days=i)).isoformat(), "close": 100 + i})
    inverse_records = []
    for i in range(30):
        inverse_records.append({"date": (base_date + dt.timedelta(days=i)).isoformat(), "close": 120 - i})
    return {
        "query": "Please run zscore, rsi and sma analysis",
        "uploaded_data": [
            {"symbol": "AAA", "records": records},
            {"symbol": "BBB", "records": inverse_records},
        ],
    }


def test_analysis_endpoint_returns_results():
    response = client.post("/analysis/", json=_sample_upload_payload())
    assert response.status_code == 200
    body = response.json()
    assert "analysis" in body
    assert set(body["tool_summaries"].keys()) == {"zscore", "rsi", "sma"}
    for images in body["images"].values():
        assert all(isinstance(img, str) and len(img) > 10 for img in images)


def test_missing_data_returns_error():
    response = client.post("/analysis/", json={"query": "test"})
    assert response.status_code == 400

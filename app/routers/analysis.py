from __future__ import annotations

import datetime as dt
import json
from typing import Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.services.agent import AgentService
from app.services.preprocess import PriceMatrix, clean_stock_data, parse_uploaded_prices
from app.services.sources import load_market_data

router = APIRouter()


class PriceRecord(BaseModel):
    date: dt.date
    close: float


class UploadedTicker(BaseModel):
    symbol: str = Field(..., description="Ticker symbol for the uploaded data")
    records: list[PriceRecord] = Field(
        default_factory=list, description="Chronological price records"
    )

    @field_validator("records")
    @classmethod
    def validate_records(cls, value: list[PriceRecord]) -> list[PriceRecord]:
        if not value:
            raise ValueError("records must contain at least one price point")
        return value


class AnalysisRequest(BaseModel):
    query: str = Field(..., description="User natural-language question")
    tickers: Optional[list[str]] = Field(
        default=None, description="Symbols to download via yfinance"
    )
    start_date: Optional[dt.date] = Field(
        default=None, description="Optional start date for downloads"
    )
    end_date: Optional[dt.date] = Field(
        default=None, description="Optional end date for downloads"
    )
    uploaded_data: Optional[list[UploadedTicker]] = Field(
        default=None, description="Optional price data supplied by the user"
    )


class AnalysisResponse(BaseModel):
    analysis: str
    tool_summaries: dict[str, str]
    images: dict[str, list[str]]


agent_service = AgentService()


@router.post("/", response_model=AnalysisResponse)
async def run_analysis(
    http_request: Request,
    request_payload: Optional[str] = Form(
        default=None, description="JSON payload for the analysis request"
    ),
    upload_file: Union[UploadFile, str, None] = File(
        None, description="optional file upload"
    ),
) -> AnalysisResponse:
    if request_payload is not None:
        try:
            request_dict = json.loads(request_payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid client payload
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}")
    else:
        try:
            request_dict = await http_request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")

    request = AnalysisRequest.model_validate(request_dict)
    raw_records: list[dict[str, object]] = []
    if isinstance(upload_file, UploadFile):
        file_bytes = await upload_file.read()
        raw_records.extend(parse_uploaded_prices(file_bytes, upload_file.filename))
    if request.uploaded_data:
        for ticker in request.uploaded_data:
            for record in ticker.records:
                raw_records.append(
                    {
                        "Date": record.date,
                        "Ticker": ticker.symbol.upper(),
                        "Close": record.close,
                    }
                )
    if not raw_records and request.tickers:
        raw_records = load_market_data(
            tickers=request.tickers,
            start=request.start_date,
            end=request.end_date,
        )
        if not raw_records:
            raise HTTPException(status_code=404, detail="No market data available for the requested tickers")
    if not raw_records:
        raise HTTPException(status_code=400, detail="Provide either tickers or uploaded_data")

    cleaned: PriceMatrix = clean_stock_data(raw_records)
    if cleaned.is_empty():
        raise HTTPException(status_code=422, detail="Unable to clean the provided stock data")

    result = agent_service.run_analysis(request.query, cleaned)

    return AnalysisResponse(
        analysis=result.summary,
        tool_summaries=result.tool_summaries,
        images=result.tool_images,
    )

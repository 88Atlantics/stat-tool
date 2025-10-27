from __future__ import annotations

import datetime as dt
from typing import Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.agent import AgentService
from app.services.preprocess import PriceMatrix, clean_stock_data, parse_uploaded_prices
from app.services.sources import load_market_data

router = APIRouter()


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


class AnalysisResponse(BaseModel):
    analysis: str
    tool_summaries: dict[str, str]
    images: dict[str, list[str]]


agent_service = AgentService()


@router.post("/", response_model=AnalysisResponse)
async def run_analysis(
    query: str = Form(..., description="User natural-language question"),
    tickers: Optional[str] = Form(
        default=None,
        description="Optional comma-separated ticker symbols to download via yfinance",
    ),
    start_date: Optional[str] = Form(
        default=None, description="Optional start date in YYYY-MM-DD format"
    ),
    end_date: Optional[str] = Form(
        default=None, description="Optional end date in YYYY-MM-DD format"
    ),
    upload_file: Union[UploadFile, str, None] = File(
        None, description="optional file upload"
    ),
) -> AnalysisResponse:
    parsed_tickers = None
    if tickers:
        symbols = [symbol.strip().upper() for symbol in tickers.split(",")]
        parsed_tickers = [symbol for symbol in symbols if symbol]
        if not parsed_tickers:
            parsed_tickers = None

    def _parse_date(value: Optional[str], field_name: str) -> Optional[dt.date]:
        if value is None or value == "":
            return None
        try:
            return dt.date.fromisoformat(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid {field_name} format. Expected YYYY-MM-DD.",
            ) from exc

    request = AnalysisRequest(
        query=query,
        tickers=parsed_tickers,
        start_date=_parse_date(start_date, "start_date"),
        end_date=_parse_date(end_date, "end_date"),
    )
    raw_records: list[dict[str, object]] = []
    if isinstance(upload_file, UploadFile):
        file_bytes = await upload_file.read()
        raw_records.extend(parse_uploaded_prices(file_bytes, upload_file.filename))
    if not raw_records and request.tickers:
        raw_records = load_market_data(
            tickers=request.tickers,
            start=request.start_date,
            end=request.end_date,
        )
        if not raw_records:
            raise HTTPException(status_code=404, detail="No market data available for the requested tickers")
    if not raw_records:
        raise HTTPException(
            status_code=400,
            detail="Provide either tickers or an uploaded price file",
        )

    cleaned: PriceMatrix = clean_stock_data(raw_records)
    if cleaned.is_empty():
        raise HTTPException(status_code=422, detail="Unable to clean the provided stock data")

    result = agent_service.run_analysis(request.query, cleaned)

    return AnalysisResponse(
        analysis=result.summary,
        tool_summaries=result.tool_summaries,
        images=result.tool_images,
    )

from __future__ import annotations

import datetime as dt
from typing import Optional

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
    upload_file: UploadFile | None = File(
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
    plan = agent_service.interpret_query(request.query)
    raw_records: list[dict[str, object]] = []
    fallback_tickers: list[str] = []
    if request.tickers:
        fallback_tickers.extend(request.tickers)
    else:
        fallback_tickers.extend(plan.tickers)
    if upload_file and upload_file.filename:
        file_bytes = await upload_file.read()
        raw_records.extend(
            parse_uploaded_prices(
                file_bytes,
                upload_file.filename,
                fallback_tickers=fallback_tickers,
            )
        )
    derived_tickers = request.tickers or list(plan.tickers)
    if not raw_records and derived_tickers:
        derived_end = request.end_date or plan.end_date or agent_service.current_date()
        derived_start = request.start_date or plan.start_date
        if derived_start is None:
            derived_start = agent_service.default_lookback(12, derived_end)
        raw_records = load_market_data(
            tickers=derived_tickers,
            start=derived_start,
            end=derived_end,
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

    result = agent_service.run_analysis(request.query, cleaned, plan=plan)

    return AnalysisResponse(
        analysis=result.summary,
        tool_summaries=result.tool_summaries,
        images=result.tool_images,
    )

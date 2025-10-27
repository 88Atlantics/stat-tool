from __future__ import annotations

from typing import List

from app.services.models import ToolResult
from app.services.preprocess import PriceMatrix
from app.services.visuals import figure_to_url, plot_lines


def _compute_rsi(prices: List[float], period: int = 14) -> List[float]:
    if len(prices) < 2:
        return [50.0 for _ in prices]

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [abs(min(delta, 0.0)) for delta in deltas]

    avg_gain = gains[0] if gains else 0.0
    avg_loss = losses[0] if losses else 0.0
    rsi_values = [50.0] * len(prices)

    for i in range(1, len(prices)):
        gain = gains[i - 1] if i - 1 < len(gains) else 0.0
        loss = losses[i - 1] if i - 1 < len(losses) else 0.0
        avg_gain = ((period - 1) * avg_gain + gain) / period
        avg_loss = ((period - 1) * avg_loss + loss) / period
        if avg_loss == 0:
            rs = float("inf")
        else:
            rs = avg_gain / avg_loss
        rsi_values[i] = 100 - (100 / (1 + rs)) if rs != float("inf") else 100.0

    return rsi_values


def analyze_rsi(data: PriceMatrix, period: int = 14) -> ToolResult:
    if data.is_empty():
        return ToolResult(name="rsi", summary="No price data provided for RSI.", images=[])

    summaries: list[str] = []
    series_map: dict[str, List[float]] = {}

    for ticker in data.tickers:
        prices = data.series[ticker]
        rsi_values = _compute_rsi(prices, period=period)
        latest = rsi_values[-1]
        if latest > 70:
            state = "overbought"
        elif latest < 30:
            state = "oversold"
        else:
            state = "neutral"
        summaries.append(f"{ticker}: {state} (RSI {latest:.1f})")
        series_map[ticker] = rsi_values

    figure = plot_lines(
        dates=[date.isoformat() for date in data.dates],
        series_map=series_map,
        title=f"{period}-Period RSI",
        y_label="RSI",
        horizontal_lines=[(70, "#d62728"), (30, "#2ca02c")],
    )
    image_url = figure_to_url(figure)

    summary = "; ".join(summaries)
    return ToolResult(name="rsi", summary=summary, images=[image_url])

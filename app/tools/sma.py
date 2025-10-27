from __future__ import annotations

from typing import List

from app.services.models import ToolResult
from app.services.preprocess import PriceMatrix
from app.services.visuals import figure_to_url, plot_lines


def _simple_moving_average(values: List[float], window: int) -> List[float]:
    averages: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        averages.append(sum(window_values) / len(window_values))
    return averages


def analyze_sma(data: PriceMatrix, windows: tuple[int, int] = (20, 50)) -> ToolResult:
    if data.is_empty():
        return ToolResult(name="sma", summary="No price data provided for moving averages.", images=[])

    short_window, long_window = windows
    summaries: list[str] = []
    images: list[str] = []

    for ticker in data.tickers:
        prices = data.series[ticker]
        short_ma = _simple_moving_average(prices, short_window)
        long_ma = _simple_moving_average(prices, long_window)

        figure = plot_lines(
            dates=[date.isoformat() for date in data.dates],
            series_map={
                f"{ticker} Close": prices,
                f"SMA {short_window}": short_ma,
                f"SMA {long_window}": long_ma,
            },
            title=f"{ticker} Close with SMAs",
            y_label="Price",
        )
        images.append(figure_to_url(figure))

        if short_ma[-1] > long_ma[-1]:
            summaries.append(f"{ticker}: bullish crossover ({short_window}>{long_window}).")
        elif short_ma[-1] < long_ma[-1]:
            summaries.append(f"{ticker}: bearish crossover ({short_window}<{long_window}).")
        else:
            summaries.append(f"{ticker}: SMAs aligned.")

    summary = " ".join(summaries)
    return ToolResult(name="sma", summary=summary, images=images)

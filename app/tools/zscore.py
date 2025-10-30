from __future__ import annotations

from itertools import combinations
from statistics import mean, pstdev
from typing import List

from app.services.models import ToolResult
from app.services.preprocess import PriceMatrix
from app.services.visuals import figure_to_payload, plot_heatmap, plot_lines


def analyze_zscore(data: PriceMatrix) -> ToolResult:
    tickers = data.tickers
    if len(tickers) < 2 or not data.dates:
        return ToolResult(
            name="zscore",
            summary="Z-score analysis requires at least two tickers.",
            images=[],
        )

    latest_scores: dict[tuple[str, str], float] = {}
    zscore_history: dict[tuple[str, str], List[float]] = {}

    for left, right in combinations(tickers, 2):
        left_series = data.series[left]
        right_series = data.series[right]
        spread = [l - r for l, r in zip(left_series, right_series)]
        spread_std = pstdev(spread)
        if spread_std == 0:
            continue
        spread_mean = mean(spread)
        z_values = [(value - spread_mean) / spread_std for value in spread]
        latest_scores[(left, right)] = z_values[-1]
        zscore_history[(left, right)] = z_values

    if not latest_scores:
        return ToolResult(
            name="zscore",
            summary="Unable to compute z-scores for the provided data.",
            images=[],
        )

    matrix = [[0.0 for _ in tickers] for _ in tickers]
    for i, left in enumerate(tickers):
        for j, right in enumerate(tickers):
            if left == right:
                continue
            if (left, right) in latest_scores:
                matrix[i][j] = latest_scores[(left, right)]
            elif (right, left) in latest_scores:
                matrix[i][j] = -latest_scores[(right, left)]

    heatmap_fig = plot_heatmap(matrix, tickers, "Latest Pairwise Z-Scores")
    heatmap_payload = figure_to_payload(heatmap_fig)

    most_extreme_pair = max(latest_scores.items(), key=lambda item: abs(item[1]))[0]
    z_series = zscore_history[most_extreme_pair]
    line_chart_fig = plot_lines(
        dates=[date.isoformat() for date in data.dates],
        series_map={f"{most_extreme_pair[0]}-{most_extreme_pair[1]}": z_series},
        title="Spread Z-Score Over Time",
        y_label="Z-Score",
        horizontal_lines=[(0, "#555555"), (2, "#d62728"), (-2, "#2ca02c")],
    )
    line_chart_payload = figure_to_payload(line_chart_fig)

    summary = (
        f"Largest divergence observed for {most_extreme_pair[0]} vs {most_extreme_pair[1]} "
        f"(z-score {latest_scores[most_extreme_pair]:.2f})."
    )

    return ToolResult(
        name="zscore",
        summary=summary,
        images=[heatmap_payload, line_chart_payload],
    )

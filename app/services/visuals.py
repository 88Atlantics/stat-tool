from __future__ import annotations

import base64
from typing import Dict, Iterable, List, Sequence, Tuple

Color = Tuple[int, int, int]


def _color_to_hex(color: Color) -> str:
    return "#" + "".join(f"{component:02x}" for component in color)


def _interpolate_color(value: float, start: Color, end: Color) -> Color:
    return tuple(int(start[i] + (end[i] - start[i]) * value) for i in range(3))


def svg_to_base64(svg: str) -> str:
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")


def build_heatmap_svg(matrix: List[List[float]], labels: Sequence[str], title: str) -> str:
    if not matrix or not labels:
        return svg_to_base64("<svg xmlns='http://www.w3.org/2000/svg'></svg>")

    size = 60
    width = size * len(labels) + 140
    height = size * len(labels) + 140
    max_abs = max(abs(value) for row in matrix for value in row) or 1

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' font-family='Arial'>",
        f"<text x='{width/2}' y='40' text-anchor='middle' font-size='20'>{title}</text>",
    ]

    start_color = (31, 119, 180)  # blue
    end_color = (214, 39, 40)  # red

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ratio = (value + max_abs) / (2 * max_abs)
            color = _interpolate_color(ratio, start_color, end_color)
            x = 80 + j * size
            y = 80 + i * size
            svg_parts.append(
                f"<rect x='{x}' y='{y}' width='{size}' height='{size}' fill='{_color_to_hex(color)}' stroke='#ffffff' stroke-width='1'/>"
            )
            svg_parts.append(
                f"<text x='{x + size/2}' y='{y + size/2 + 5}' text-anchor='middle' font-size='14' fill='#ffffff'>{value:.2f}</text>"
            )

    for idx, label in enumerate(labels):
        x = 80 + idx * size + size / 2
        svg_parts.append(
            f"<text x='{x}' y='{height - 40}' text-anchor='middle' font-size='14'>{label}</text>"
        )
        y = 80 + idx * size + size / 2
        svg_parts.append(
            f"<text x='40' y='{y + 5}' text-anchor='end' font-size='14'>{label}</text>"
        )

    svg_parts.append("</svg>")
    return svg_to_base64("".join(svg_parts))


def build_line_chart_svg(
    dates: Sequence[str],
    series_map: Dict[str, Sequence[float]],
    title: str,
    y_label: str,
    horizontal_lines: Iterable[Tuple[float, str]] | None = None,
) -> str:
    if not dates:
        return svg_to_base64("<svg xmlns='http://www.w3.org/2000/svg'></svg>")

    width = 820
    height = 360
    margin_left = 70
    margin_bottom = 60
    margin_top = 60
    margin_right = 30
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for series in series_map.values() for value in series]
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    def to_x(idx: int) -> float:
        if len(dates) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * idx) / (len(dates) - 1)

    def to_y(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min) * plot_height)

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' font-family='Arial'>",
        f"<text x='{width/2}' y='30' text-anchor='middle' font-size='20'>{title}</text>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#333'/>",
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#333'/>",
        f"<text x='{margin_left - 40}' y='{margin_top + plot_height/2}' transform='rotate(-90 {margin_left - 40},{margin_top + plot_height/2})' text-anchor='middle' font-size='14'>{y_label}</text>",
    ]

    for idx in range(5):
        y_value = y_min + (y_max - y_min) * idx / 4
        y_pos = to_y(y_value)
        svg_parts.append(
            f"<line x1='{margin_left}' y1='{y_pos}' x2='{width - margin_right}' y2='{y_pos}' stroke='#e0e0e0' stroke-dasharray='4 4'/>"
        )
        svg_parts.append(
            f"<text x='{margin_left - 10}' y='{y_pos + 4}' text-anchor='end' font-size='12'>{y_value:.2f}</text>"
        )

    if horizontal_lines:
        for value, color in horizontal_lines:
            y_pos = to_y(value)
            svg_parts.append(
                f"<line x1='{margin_left}' y1='{y_pos}' x2='{width - margin_right}' y2='{y_pos}' stroke='{color}' stroke-dasharray='6 3' stroke-width='1.5'/>"
            )

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    for idx, (label, series) in enumerate(series_map.items()):
        color = palette[idx % len(palette)]
        points = " ".join(f"{to_x(i)},{to_y(value)}" for i, value in enumerate(series))
        svg_parts.append(
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>"
        )
        last_x = to_x(len(series) - 1)
        last_y = to_y(series[-1])
        svg_parts.append(
            f"<circle cx='{last_x}' cy='{last_y}' r='3' fill='{color}'/>"
        )
        svg_parts.append(
            f"<text x='{width - margin_right - 10}' y='{margin_top + 20 + idx * 18}' text-anchor='end' font-size='13' fill='{color}'>{label}</text>"
        )

    for i, label in enumerate(_select_labels(dates)):
        x_pos = to_x(label.index)
        svg_parts.append(
            f"<text x='{x_pos}' y='{height - margin_bottom + 20}' text-anchor='middle' font-size='12'>{label.text}</text>"
        )

    svg_parts.append("</svg>")
    return svg_to_base64("".join(svg_parts))


class _AxisLabel:
    def __init__(self, index: int, text: str) -> None:
        self.index = index
        self.text = text


def _select_labels(dates: Sequence[str]) -> List[_AxisLabel]:
    if not dates:
        return []
    if len(dates) <= 4:
        return [_AxisLabel(idx, label) for idx, label in enumerate(dates)]
    indices = [0, len(dates) // 2, len(dates) - 1]
    seen = set()
    labels = []
    for idx in indices:
        label = dates[idx]
        if label not in seen:
            labels.append(_AxisLabel(idx, label))
            seen.add(label)
    return labels

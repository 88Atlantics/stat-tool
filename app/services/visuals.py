from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote

try:  # pragma: no cover - matplotlib optional during testing
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - fallback when matplotlib missing
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in tests
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:  # pragma: no cover - optional dependency in tests
    BlobServiceClient = None  # type: ignore[assignment]
    ContentSettings = None  # type: ignore[assignment]


@dataclass(slots=True)
class VisualConfig:
    connection_string: str | None
    container_name: str | None

    @classmethod
    def from_env(cls) -> "VisualConfig":
        return cls(
            connection_string=os.getenv("AZURE_BLOB_CONNECTION_STRING"),
            container_name=os.getenv("AZURE_BLOB_CONTAINER"),
        )


def _figure_to_png_bytes(fig) -> bytes:
    if plt is None or not hasattr(fig, "savefig"):
        raise RuntimeError("Matplotlib is not available")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _upload_to_azure(content: bytes, config: VisualConfig) -> str | None:
    if not config.connection_string or not config.container_name:
        return None
    if BlobServiceClient is None:
        return None

    blob_service = BlobServiceClient.from_connection_string(config.connection_string)
    blob_client = blob_service.get_blob_client(
        container=config.container_name,
        blob=f"analysis/{os.urandom(8).hex()}.png",
    )
    blob_client.upload_blob(  # type: ignore[union-attr]
        content,
        overwrite=True,
        content_settings=ContentSettings(content_type="image/png"),  # type: ignore[call-arg]
    )
    return blob_client.url  # type: ignore[return-value]


def figure_to_url(fig, config: VisualConfig | None = None) -> str:
    cfg = config or VisualConfig.from_env()
    if plt is None or not hasattr(fig, "savefig"):
        text = getattr(fig, "payload", "Chart unavailable")
        return f"data:text/plain;charset=utf-8,{quote(str(text))}"
    png_bytes = _figure_to_png_bytes(fig)
    azure_url = _upload_to_azure(png_bytes, cfg)
    if azure_url:
        return azure_url
    base64_image = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"


def plot_heatmap(matrix: Sequence[Sequence[float]], labels: Sequence[str], title: str):
    if plt is None:  # pragma: no cover - fallback text output when matplotlib missing
        return type("DummyFigure", (), {"payload": f"{title}: heatmap unavailable"})()
    fig, ax = plt.subplots(figsize=(0.6 * max(len(labels), 1) + 3, 0.6 * max(len(labels), 1) + 3))
    heatmap = ax.imshow(matrix, cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", va="center", ha="center", color="black")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_lines(
    dates: Sequence[str],
    series_map: Mapping[str, Sequence[float]],
    title: str,
    y_label: str,
    horizontal_lines: Iterable[tuple[float, str]] | None = None,
):
    if plt is None:  # pragma: no cover - fallback text output when matplotlib missing
        readable = ", ".join(f"{label}" for label in series_map.keys())
        return type(
            "DummyFigure",
            (),
            {"payload": f"{title}: {readable} (plot unavailable)"},
        )()
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in series_map.items():
        ax.plot(dates, series, label=label)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Date")
    if horizontal_lines:
        for value, color in horizontal_lines:
            ax.axhline(y=value, color=color, linestyle="--", linewidth=1)
    if len(series_map) > 1:
        ax.legend()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers.analysis import router as analysis_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Investment Statistics API",
        description=(
            "An agent-driven FastAPI service that interprets investment queries, "
            "runs quantitative tools, and returns chart-ready insights."
        ),
        version="0.1.0",
    )
    static_root = Path(__file__).resolve().parent / "static"
    static_root.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_root), name="static")
    app.include_router(analysis_router, prefix="/analysis", tags=["analysis"])

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Investment statistics agent is ready."}

    return app


app = create_app()

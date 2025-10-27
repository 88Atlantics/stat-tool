from fastapi import FastAPI

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
    app.include_router(analysis_router, prefix="/analysis", tags=["analysis"])

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Investment statistics agent is ready."}

    return app


app = create_app()

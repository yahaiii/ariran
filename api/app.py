from fastapi import FastAPI

from api.routes.incidents import router as incidents_router
from api.routes.pipeline import router as pipeline_router


def create_app() -> FastAPI:
    app = FastAPI(title="ariran API", version="0.1.0")
    app.include_router(incidents_router, prefix="/incidents", tags=["incidents"])
    app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
    return app


app = create_app()

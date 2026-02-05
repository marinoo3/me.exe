from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config import settings
from .routers import api
from app.services import RagService




@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup logic ---
    app.state.rag_service = RagService()

    yield

    # --- shutdown logic ---
    del app.state.rag_service


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


@app.get("/ping", tags=["ping"])
async def ping():
    """Simple health check endpoint."""
    return {"status": "ok"}


# Include routers
app.include_router(api.router)
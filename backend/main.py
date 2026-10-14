from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import get_settings
from backend.db.session import engine


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    description="Intelligent Model and Reasoning Router",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "name": "BrainRoute",
        "version": "0.1.0",
        "message": "Intelligent AI Model Router",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "brainroute",
    }


@app.get("/health/db")
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "connected",
        }

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed",
        ) from exc

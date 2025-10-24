from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Basic logging configuration for debugging (works with uvicorn reload)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("ai").setLevel(logging.INFO)
logging.getLogger("ai.ai_service").setLevel(logging.INFO)
logging.getLogger("ai.data_service").setLevel(logging.INFO)
logging.getLogger("ai.api").setLevel(logging.INFO)

from .routes import api

app = FastAPI(title="AI Excel Assistant API", version="0.1.0")

# Dev CORS - open for local dev. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173", "http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api.router)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "ai-excel-backend"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

"""FastAPI entry point for the Ranjana Lipi coaching backend."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.characters import router as characters_router
from app.api.practice import router as practice_router
from app.api.progress import router as progress_router
from app.core.database import SessionLocal
from app.services.characters import seed_default_characters


app = FastAPI(title="Ranjana Lipi Coach API")

# Development only: allow all origins while the Expo app and backend are local.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(characters_router)
app.include_router(practice_router)
app.include_router(progress_router)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_DIR = BACKEND_ROOT / "ml" / "processed" / "references"
GENERAL_REFERENCES_DIR = BACKEND_ROOT / "ml" / "processed_general" / "references"
UPLOADS_DIR = BACKEND_ROOT / "uploads"

if REFERENCES_DIR.is_dir():
    app.mount("/references", StaticFiles(directory=str(REFERENCES_DIR)), name="references")
if GENERAL_REFERENCES_DIR.is_dir():
    app.mount(
        "/references_general",
        StaticFiles(directory=str(GENERAL_REFERENCES_DIR)),
        name="references_general",
    )
if UPLOADS_DIR.is_dir():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


@app.on_event("startup")
def seed_characters_on_startup() -> None:
    db = SessionLocal()
    try:
        seed_default_characters(db)
    finally:
        db.close()


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ranjana-lipi-coach"}

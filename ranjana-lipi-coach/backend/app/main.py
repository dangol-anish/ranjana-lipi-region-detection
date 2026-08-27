"""FastAPI entry point for the Ranjana Lipi coaching backend."""

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.characters import router as characters_router
from app.api.practice import router as practice_router
from app.api.profile import router as profile_router
from app.api.progress import router as progress_router
from app.api.recommendations import router as recommendations_router
from app.core.database import SessionLocal
from app.services.characters import seed_default_characters


app = FastAPI(title="Ranjana Lipi Handwriting Learner API")

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
app.include_router(profile_router)
app.include_router(progress_router)
app.include_router(recommendations_router)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
REFERENCES_DIR = BACKEND_ROOT / "ml" / "processed" / "references"
GENERAL_REFERENCES_DIR = BACKEND_ROOT / "ml" / "processed_general" / "references"
RAW_REFERENCES_DIR = PROJECT_ROOT / "data" / "Reference"
DISPLAY_GLYPHS_DIR = BACKEND_ROOT / "ml" / "display_glyphs"
STRUCTURE_VALIDATION_DIR = PROJECT_ROOT / "data" / "StructureValidation"
UPLOADS_DIR = BACKEND_ROOT / "uploads"
DEMO_CANVAS_DIR = PROJECT_ROOT / "data" / "Canvas"
SPECIALIZED_DISPLAY_CLASSES = {
    "a",
    "aa",
    "cha",
    "da",
    "dda",
    "ddha",
    "ga",
    "gha",
    "ja",
    "jha",
    "ka",
    "ma",
    "nna",
    "ta",
    "ya",
}
DISPLAY_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
NO_CACHE_HEADERS = {"Cache-Control": "no-store, max-age=0"}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

if REFERENCES_DIR.is_dir():
    app.mount("/references", StaticFiles(directory=str(REFERENCES_DIR)), name="references")
if GENERAL_REFERENCES_DIR.is_dir():
    app.mount(
        "/references_general",
        StaticFiles(directory=str(GENERAL_REFERENCES_DIR)),
        name="references_general",
    )
if RAW_REFERENCES_DIR.is_dir():
    app.mount("/reference_photos", StaticFiles(directory=str(RAW_REFERENCES_DIR)), name="reference_photos")


@app.get("/display_glyphs/{filename}")
def display_glyph(filename: str) -> FileResponse:
    character_name = Path(filename).stem
    if character_name in SPECIALIZED_DISPLAY_CLASSES:
        good_dir = STRUCTURE_VALIDATION_DIR / character_name / "good"
        for extension in DISPLAY_IMAGE_EXTENSIONS:
            candidate = good_dir / f"{character_name}_good_01{extension}"
            if candidate.is_file():
                return FileResponse(str(candidate), headers=NO_CACHE_HEADERS)

    fallback = DISPLAY_GLYPHS_DIR / f"{character_name}.png"
    if fallback.is_file():
        return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)

    raise HTTPException(status_code=404, detail="Display glyph not found.")


if DEMO_CANVAS_DIR.is_dir():
    app.mount("/demo_canvas", StaticFiles(directory=str(DEMO_CANVAS_DIR)), name="demo_canvas")
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

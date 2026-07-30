"""FastAPI entry point for the Ranjana Lipi coaching backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router


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


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ranjana-lipi-coach"}

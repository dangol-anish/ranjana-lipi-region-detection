"""Practice attempt persistence and progress updates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.attempt import Attempt, PracticeMode
from app.models.character import Character
from app.models.progress import UserCharacterProgress
from app.models.user import User


MASTERED_SCORE_THRESHOLD = 95.0


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def save_attempt_image(user: User, character: Character, normalized_image: np.ndarray) -> str:
    output_dir = backend_root() / "uploads" / "attempts" / str(user.id) / character.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid4().hex}.png"
    cv2.imwrite(str(output_path), (normalized_image * 255).astype("uint8"))
    return str(output_path.relative_to(backend_root()))


def update_progress(
    db: Session,
    user: User,
    character: Character,
    score: float | None,
) -> UserCharacterProgress:
    progress = db.scalar(
        select(UserCharacterProgress).where(
            UserCharacterProgress.user_id == user.id,
            UserCharacterProgress.character_id == character.id,
        )
    )
    if progress is None:
        progress = UserCharacterProgress(
            user_id=user.id,
            character_id=character.id,
            attempts_count=0,
            best_score=None,
            mastered=False,
        )
        db.add(progress)

    progress.attempts_count += 1
    progress.last_practiced_at = datetime.now(timezone.utc)
    if score is not None:
        progress.best_score = score if progress.best_score is None else max(progress.best_score, score)
        progress.mastered = progress.best_score >= MASTERED_SCORE_THRESHOLD
    return progress


def create_attempt_with_progress(
    db: Session,
    user: User,
    character: Character,
    mode: PracticeMode,
    image_path: str,
    overall_score: float | None,
    region_feedback: dict,
) -> tuple[Attempt, UserCharacterProgress]:
    attempt = Attempt(
        user_id=user.id,
        character_id=character.id,
        mode=mode,
        image_path=image_path,
        overall_score=overall_score,
        region_feedback=region_feedback,
    )
    db.add(attempt)
    progress = update_progress(db, user, character, overall_score)
    db.commit()
    db.refresh(attempt)
    db.refresh(progress)
    return attempt, progress

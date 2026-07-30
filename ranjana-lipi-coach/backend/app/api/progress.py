"""Progress dashboard endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.attempt import Attempt
from app.models.character import Character
from app.models.progress import UserCharacterProgress
from app.models.user import User
from app.schemas.attempt import AttemptOut
from app.schemas.character import CharacterOut
from app.schemas.progress import CharacterProgressDetail, ProgressDashboardItem, UserCharacterProgressOut


router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("", response_model=list[ProgressDashboardItem])
def get_progress_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgressDashboardItem]:
    characters = list(db.scalars(select(Character).order_by(Character.id)).all())
    progress_rows = db.scalars(
        select(UserCharacterProgress).where(UserCharacterProgress.user_id == current_user.id)
    ).all()
    progress_by_character = {
        progress.character_id: progress
        for progress in progress_rows
    }
    return [
        ProgressDashboardItem(
            character=CharacterOut.model_validate(character),
            progress=(
                UserCharacterProgressOut.model_validate(progress_by_character[character.id])
                if character.id in progress_by_character
                else None
            ),
        )
        for character in characters
    ]


@router.get("/{character_id}", response_model=CharacterProgressDetail)
def get_character_progress(
    character_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CharacterProgressDetail:
    character = db.get(Character, character_id)
    if character is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found",
        )

    progress = db.scalar(
        select(UserCharacterProgress).where(
            UserCharacterProgress.user_id == current_user.id,
            UserCharacterProgress.character_id == character.id,
        )
    )
    attempts = list(
        db.scalars(
            select(Attempt)
            .where(
                Attempt.user_id == current_user.id,
                Attempt.character_id == character.id,
            )
            .order_by(Attempt.created_at.desc())
        ).all()
    )

    return CharacterProgressDetail(
        character=CharacterOut.model_validate(character),
        progress=UserCharacterProgressOut.model_validate(progress) if progress else None,
        attempts=[AttemptOut.model_validate(attempt) for attempt in attempts],
    )

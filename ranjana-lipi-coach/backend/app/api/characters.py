"""Character catalog endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.character import Character
from app.models.user import User
from app.schemas.character import CharacterOut


router = APIRouter(prefix="/characters", tags=["characters"])


@router.get("", response_model=list[CharacterOut])
def list_characters(
    _current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Character]:
    return list(db.scalars(select(Character).order_by(Character.id)).all())

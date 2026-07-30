"""Import all SQLAlchemy models so Alembic can discover Base.metadata."""

from app.models.attempt import Attempt, PracticeMode
from app.models.character import Character
from app.models.progress import UserCharacterProgress
from app.models.user import User

__all__ = [
    "Attempt",
    "Character",
    "PracticeMode",
    "User",
    "UserCharacterProgress",
]

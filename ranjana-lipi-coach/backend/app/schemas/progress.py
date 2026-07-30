"""User progress schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserCharacterProgressBase(BaseModel):
    character_id: int
    attempts_count: int = 0
    best_score: float | None = None
    last_practiced_at: datetime | None = None
    mastered: bool = False


class UserCharacterProgressCreate(UserCharacterProgressBase):
    user_id: int


class UserCharacterProgressOut(UserCharacterProgressBase):
    id: int
    user_id: int

    model_config = ConfigDict(from_attributes=True)

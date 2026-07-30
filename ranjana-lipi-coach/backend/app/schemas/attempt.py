"""Practice attempt schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.attempt import PracticeMode


class AttemptBase(BaseModel):
    character_id: int
    mode: PracticeMode
    image_path: str = Field(..., max_length=500)
    overall_score: float | None = None
    region_feedback: dict[str, Any] | None = None


class AttemptCreate(AttemptBase):
    pass


class AttemptOut(AttemptBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

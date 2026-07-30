"""Practice attempt model."""

import enum
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class PracticeMode(str, enum.Enum):
    app_suggested = "app_suggested"
    free_practice = "free_practice"
    assessment = "assessment"


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    character_id: Mapped[int] = mapped_column(
        ForeignKey("characters.id"),
        nullable=False,
        index=True,
    )
    mode: Mapped[PracticeMode] = mapped_column(
        Enum(PracticeMode, name="practice_mode"),
        nullable=False,
    )
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    region_feedback: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="attempts")
    character = relationship("Character", back_populates="attempts")

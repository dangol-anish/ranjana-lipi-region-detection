"""Aggregated user progress per character."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class UserCharacterProgress(Base):
    __tablename__ = "user_character_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "character_id", name="uq_user_character_progress"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    character_id: Mapped[int] = mapped_column(
        ForeignKey("characters.id"),
        nullable=False,
        index=True,
    )
    attempts_count: Mapped[int] = mapped_column(default=0, nullable=False)
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_practiced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    mastered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="progress")
    character = relationship("Character", back_populates="progress")

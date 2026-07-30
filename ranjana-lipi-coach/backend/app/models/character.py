"""Ranjana character model."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    display_label: Mapped[str] = mapped_column(String(120), nullable=False)
    region_grid_rows: Mapped[int] = mapped_column(default=3, nullable=False)
    region_grid_cols: Mapped[int] = mapped_column(default=3, nullable=False)

    attempts = relationship("Attempt", back_populates="character", cascade="all, delete-orphan")
    progress = relationship(
        "UserCharacterProgress",
        back_populates="character",
        cascade="all, delete-orphan",
    )

"""User profile and practice heatmap schemas."""

from datetime import date, datetime

from pydantic import BaseModel

from app.schemas.user import UserOut


class PracticeHeatmapDay(BaseModel):
    date: date
    attempts_count: int
    average_score: float | None = None
    best_score: float | None = None


class UserProfileStats(BaseModel):
    total_attempts: int
    practiced_characters: int
    mastered_characters: int
    average_score: float | None = None
    best_score: float | None = None
    current_streak_days: int
    longest_streak_days: int


class UserProfileOut(BaseModel):
    user: UserOut
    stats: UserProfileStats
    heatmap: list[PracticeHeatmapDay]
    generated_at: datetime

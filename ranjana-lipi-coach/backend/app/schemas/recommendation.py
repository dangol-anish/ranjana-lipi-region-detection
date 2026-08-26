"""Adaptive practice recommendation schemas."""

from pydantic import BaseModel

from app.schemas.character import CharacterOut


class RecommendationSignal(BaseModel):
    recent_average_score: float | None = None
    attempts_count: int = 0
    best_score: float | None = None
    last_practiced_at: str | None = None
    weakest_region: str | None = None
    weak_region_repeat_count: int = 0
    review_interval_hours: float = 0.0
    elapsed_hours: float | None = None
    due_ratio: float = 0.0
    mastered: bool = False


class PracticeRecommendation(BaseModel):
    character: CharacterOut
    priority_score: float
    reason: str
    signals: RecommendationSignal


class PracticeRecommendationResponse(BaseModel):
    recommendations: list[PracticeRecommendation]
    algorithm: str
    formula: str

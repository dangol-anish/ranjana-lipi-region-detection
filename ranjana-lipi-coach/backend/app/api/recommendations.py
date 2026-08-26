"""Adaptive practice recommendation endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.recommendation import PracticeRecommendationResponse
from app.services.recommendations import build_practice_recommendations


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/practice", response_model=PracticeRecommendationResponse)
def get_practice_recommendations(
    limit: int = Query(default=5, ge=1, le=15),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PracticeRecommendationResponse:
    return PracticeRecommendationResponse(
        recommendations=build_practice_recommendations(db, current_user, limit=limit),
        algorithm="adaptive_weighted_priority",
        formula=(
            "priority = score_gap*0.45 + due_bonus + repeated_region_bonus "
            "+ new_character_bonus - mastered_penalty"
        ),
    )

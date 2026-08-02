"""Practice attempt endpoints."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.attempt import Attempt, PracticeMode
from app.models.user import User
from app.schemas.attempt import AttemptOut, PracticeAttemptResponse
from app.services.attempts import create_attempt_with_progress, save_attempt_image
from app.services.characters import get_character_by_id
from ml.inference.service import RanjanaInferenceService, inference_service


router = APIRouter(prefix="/practice", tags=["practice"])


def get_inference_service() -> RanjanaInferenceService:
    return inference_service


@router.get("/attempts", response_model=list[AttemptOut])
def list_practice_attempts(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Attempt]:
    return list(
        db.scalars(
            select(Attempt)
            .where(Attempt.user_id == current_user.id)
            .order_by(Attempt.created_at.desc())
            .limit(limit)
        ).all()
    )


@router.post("/attempt", response_model=PracticeAttemptResponse)
async def create_practice_attempt(
    character_id: int = Form(...),
    mode: PracticeMode = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    ml_service: RanjanaInferenceService = Depends(get_inference_service),
) -> PracticeAttemptResponse:
    character = get_character_by_id(db, character_id)
    if character is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty",
        )

    try:
        analysis = ml_service.analyze_practice_attempt(
            image_bytes=image_bytes,
            target_class=character.name,
            rows=character.region_grid_rows,
            cols=character.region_grid_cols,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    feedback = analysis["feedback"]
    image_path = save_attempt_image(current_user, character, analysis["normalized"])
    attempt, _progress = create_attempt_with_progress(
        db=db,
        user=current_user,
        character=character,
        mode=mode,
        image_path=image_path,
        overall_score=feedback["overall_score"],
        region_feedback=feedback,
    )

    return PracticeAttemptResponse(
        attempt=AttemptOut.model_validate(attempt),
        overall_score=feedback["overall_score"],
        region_feedback=feedback,
    )

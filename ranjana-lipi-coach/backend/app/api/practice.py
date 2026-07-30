"""Practice attempt endpoints."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.attempt import PracticeMode
from app.models.user import User
from app.schemas.attempt import AttemptOut, PracticeAttemptResponse
from app.services.attempts import create_attempt_with_progress, save_attempt_image
from app.services.characters import get_character_by_id
from ml.inference.pipeline import analyze_attempt


router = APIRouter(prefix="/practice", tags=["practice"])


@router.post("/attempt", response_model=PracticeAttemptResponse)
async def create_practice_attempt(
    character_id: int = Form(...),
    mode: PracticeMode = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
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
        analysis = analyze_attempt(
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

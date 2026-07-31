"""User profile endpoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.attempt import Attempt
from app.models.progress import UserCharacterProgress
from app.models.user import User
from app.schemas.profile import PracticeHeatmapDay, UserProfileOut, UserProfileStats
from app.schemas.user import UserOut


router = APIRouter(prefix="/profile", tags=["profile"])


def _date_key(value: datetime) -> date:
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(timezone.utc).date()


def _streaks(practiced_days: set[date], today: date) -> tuple[int, int]:
    current = 0
    cursor = today
    while cursor in practiced_days:
        current += 1
        cursor -= timedelta(days=1)

    longest = 0
    running = 0
    previous: date | None = None
    for day in sorted(practiced_days):
        if previous is None or day == previous + timedelta(days=1):
            running += 1
        else:
            running = 1
        longest = max(longest, running)
        previous = day

    return current, longest


@router.get("/me", response_model=UserProfileOut)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfileOut:
    attempts = list(
        db.scalars(
            select(Attempt)
            .where(Attempt.user_id == current_user.id)
            .order_by(Attempt.created_at.asc())
        ).all()
    )
    progress_rows = list(
        db.scalars(
            select(UserCharacterProgress).where(UserCharacterProgress.user_id == current_user.id)
        ).all()
    )

    scores = [attempt.overall_score for attempt in attempts if attempt.overall_score is not None]
    attempts_by_day: dict[date, list[float | None]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_day[_date_key(attempt.created_at)].append(attempt.overall_score)

    today = datetime.now(timezone.utc).date()
    practiced_days = set(attempts_by_day)
    current_streak, longest_streak = _streaks(practiced_days, today)

    heatmap_days: list[PracticeHeatmapDay] = []
    for offset in range(89, -1, -1):
        day = today - timedelta(days=offset)
        day_scores = [score for score in attempts_by_day.get(day, []) if score is not None]
        heatmap_days.append(
            PracticeHeatmapDay(
                date=day,
                attempts_count=len(attempts_by_day.get(day, [])),
                average_score=(sum(day_scores) / len(day_scores) if day_scores else None),
                best_score=(max(day_scores) if day_scores else None),
            )
        )

    mastered_count = sum(1 for progress in progress_rows if progress.mastered)
    practiced_characters = sum(1 for progress in progress_rows if progress.attempts_count > 0)

    return UserProfileOut(
        user=UserOut.model_validate(current_user),
        stats=UserProfileStats(
            total_attempts=len(attempts),
            practiced_characters=practiced_characters,
            mastered_characters=mastered_count,
            average_score=(sum(scores) / len(scores) if scores else None),
            best_score=(max(scores) if scores else None),
            current_streak_days=current_streak,
            longest_streak_days=longest_streak,
        ),
        heatmap=heatmap_days,
        generated_at=datetime.now(timezone.utc),
    )

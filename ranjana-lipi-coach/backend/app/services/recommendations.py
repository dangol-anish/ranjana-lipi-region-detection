"""Adaptive practice recommendation service."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.attempt import Attempt
from app.models.character import Character
from app.models.progress import UserCharacterProgress
from app.models.user import User
from app.schemas.character import CharacterOut
from app.schemas.recommendation import PracticeRecommendation, RecommendationSignal


RECENT_ATTEMPT_LIMIT = 8


def _hours_since(value: datetime | None, now: datetime) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0.0, (now - value).total_seconds() / 3600.0)


def _review_interval_hours(progress: UserCharacterProgress | None) -> float:
    if progress is None or progress.attempts_count <= 0 or progress.best_score is None:
        return 0.0

    attempts = max(1, progress.attempts_count)
    score = progress.best_score
    if score < 70:
        return 0.25
    if score < 85:
        return min(24.0, 4.0 * attempts)
    if score < 95:
        return min(24.0 * 7.0, 24.0 * (1.7 ** (attempts - 1)))
    return min(24.0 * 30.0, 24.0 * 3.0 * (2.3 ** (attempts - 1)))


def _problem_region(feedback: dict[str, Any] | None) -> str | None:
    if not feedback:
        return None
    if feedback.get("wrong_character"):
        return "wrong character"
    if feedback.get("insufficient_input"):
        return "insufficient input"

    broad = feedback.get("broad_bands")
    if isinstance(broad, dict):
        problem_regions = broad.get("problem_regions")
        if isinstance(problem_regions, list) and problem_regions:
            first = problem_regions[0]
            if isinstance(first, dict) and first.get("region"):
                return str(first["region"])

    problem_regions = feedback.get("problem_regions")
    if isinstance(problem_regions, list) and problem_regions:
        first = problem_regions[0]
        if isinstance(first, dict) and first.get("region"):
            return str(first["region"])
    return None


def _recent_attempts_by_character(db: Session, user: User, character_ids: list[int]) -> dict[int, list[Attempt]]:
    attempts = list(
        db.scalars(
            select(Attempt)
            .where(
                Attempt.user_id == user.id,
                Attempt.character_id.in_(character_ids),
            )
            .order_by(Attempt.character_id, Attempt.created_at.desc())
        ).all()
    )
    grouped: dict[int, list[Attempt]] = {character_id: [] for character_id in character_ids}
    for attempt in attempts:
        bucket = grouped.setdefault(attempt.character_id, [])
        if len(bucket) < RECENT_ATTEMPT_LIMIT:
            bucket.append(attempt)
    return grouped


def _recommendation_reason(
    progress: UserCharacterProgress | None,
    recent_average: float | None,
    weakest_region: str | None,
    weak_region_repeat_count: int,
    due_ratio: float,
) -> str:
    if progress is None or progress.attempts_count == 0:
        return "New character: start building practice history."
    if weakest_region and weak_region_repeat_count >= 2:
        return f"Repeated {weakest_region} issue across recent attempts."
    if recent_average is not None and recent_average < 70:
        return "Recent scores are low, so this needs quick review."
    if due_ratio >= 1.0:
        return "Due for spaced review."
    if progress.mastered:
        return "Mastered, but kept as a lower-priority review."
    return "Useful upcoming review based on recent progress."


def build_practice_recommendations(
    db: Session,
    user: User,
    limit: int = 5,
) -> list[PracticeRecommendation]:
    now = datetime.now(timezone.utc)
    characters = list(db.scalars(select(Character).order_by(Character.id)).all())
    progress_rows = list(
        db.scalars(
            select(UserCharacterProgress).where(UserCharacterProgress.user_id == user.id)
        ).all()
    )
    progress_by_character = {progress.character_id: progress for progress in progress_rows}
    recent_by_character = _recent_attempts_by_character(db, user, [character.id for character in characters])

    recommendations: list[PracticeRecommendation] = []
    for index, character in enumerate(characters):
        progress = progress_by_character.get(character.id)
        recent_attempts = recent_by_character.get(character.id, [])
        recent_scores = [
            float(attempt.overall_score)
            for attempt in recent_attempts
            if attempt.overall_score is not None
        ]
        recent_average = sum(recent_scores) / len(recent_scores) if recent_scores else None
        regions = [
            region
            for region in (_problem_region(attempt.region_feedback) for attempt in recent_attempts)
            if region
        ]
        region_counts = Counter(regions)
        weakest_region, weak_repeat = region_counts.most_common(1)[0] if region_counts else (None, 0)

        interval_hours = _review_interval_hours(progress)
        elapsed_hours = _hours_since(progress.last_practiced_at if progress else None, now)
        due_ratio = 4.0 if elapsed_hours is None else (4.0 if interval_hours == 0 else elapsed_hours / interval_hours)
        score_gap = 100.0 - (recent_average if recent_average is not None else 55.0)
        new_bonus = 18.0 if progress is None or progress.attempts_count == 0 else 0.0
        repeated_region_bonus = min(24.0, weak_repeat * 8.0)
        due_bonus = min(30.0, due_ratio * 10.0)
        mastered_penalty = 16.0 if progress and progress.mastered else 0.0
        tie_breaker = max(0.0, 1.0 - index * 0.001)
        priority = score_gap * 0.45 + due_bonus + repeated_region_bonus + new_bonus - mastered_penalty + tie_breaker

        recommendations.append(
            PracticeRecommendation(
                character=CharacterOut.model_validate(character),
                priority_score=round(priority, 3),
                reason=_recommendation_reason(progress, recent_average, weakest_region, weak_repeat, due_ratio),
                signals=RecommendationSignal(
                    recent_average_score=round(recent_average, 3) if recent_average is not None else None,
                    attempts_count=progress.attempts_count if progress else 0,
                    best_score=progress.best_score if progress else None,
                    last_practiced_at=progress.last_practiced_at.isoformat() if progress and progress.last_practiced_at else None,
                    weakest_region=weakest_region,
                    weak_region_repeat_count=weak_repeat,
                    review_interval_hours=round(interval_hours, 3),
                    elapsed_hours=round(elapsed_hours, 3) if elapsed_hours is not None else None,
                    due_ratio=round(due_ratio, 3),
                    mastered=bool(progress.mastered) if progress else False,
                ),
            )
        )

    return sorted(recommendations, key=lambda item: item.priority_score, reverse=True)[:limit]

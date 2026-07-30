"""Character catalog helpers."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.character import Character


DEFAULT_CHARACTERS = (
    {"name": "aa", "display_label": "Aa", "region_grid_rows": 3, "region_grid_cols": 3},
    {"name": "a", "display_label": "A", "region_grid_rows": 3, "region_grid_cols": 3},
    {"name": "ka", "display_label": "Ka", "region_grid_rows": 3, "region_grid_cols": 3},
    {"name": "da", "display_label": "Da", "region_grid_rows": 3, "region_grid_cols": 3},
    {"name": "dda", "display_label": "Dda", "region_grid_rows": 3, "region_grid_cols": 3},
)


def seed_default_characters(db: Session) -> None:
    existing_names = set(db.scalars(select(Character.name)).all())
    for item in DEFAULT_CHARACTERS:
        if item["name"] not in existing_names:
            db.add(Character(**item))
    db.commit()


def get_character_by_id(db: Session, character_id: int) -> Character | None:
    return db.get(Character, character_id)

"""Character catalog helpers."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.character import Character


GENERAL_CHARACTER_NAMES = (
    "a",
    "aa",
    "ah",
    "ai",
    "am",
    "au",
    "ba",
    "bha",
    "ca",
    "cha",
    "da",
    "dda",
    "ddha",
    "dha",
    "e",
    "eight",
    "five",
    "four",
    "ga",
    "gha",
    "gyan",
    "ha",
    "i",
    "ii",
    "ja",
    "jha",
    "ka",
    "kha",
    "ksa",
    "la",
    "lu",
    "luu",
    "ma",
    "na",
    "nine",
    "nna",
    "nnna",
    "nya",
    "o",
    "one",
    "pa",
    "pha",
    "ra",
    "ri",
    "rii",
    "sa",
    "saa",
    "seven",
    "sha",
    "six",
    "ta",
    "tha",
    "three",
    "tra",
    "tta",
    "ttha",
    "two",
    "u",
    "uu",
    "wo",
    "ya",
    "zero",
)

DEFAULT_CHARACTERS = tuple(
    {
        "name": name,
        "display_label": name.upper() if name in {"a", "i", "o", "u", "e"} else name.title(),
        "region_grid_rows": 3,
        "region_grid_cols": 3,
    }
    for name in GENERAL_CHARACTER_NAMES
)


def seed_default_characters(db: Session) -> None:
    existing_names = set(db.scalars(select(Character.name)).all())
    for item in DEFAULT_CHARACTERS:
        if item["name"] not in existing_names:
            db.add(Character(**item))
    db.commit()


def get_character_by_id(db: Session, character_id: int) -> Character | None:
    return db.get(Character, character_id)

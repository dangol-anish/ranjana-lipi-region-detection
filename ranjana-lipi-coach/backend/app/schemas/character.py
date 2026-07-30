"""Character schemas."""

from pydantic import BaseModel, ConfigDict, Field


class CharacterBase(BaseModel):
    name: str = Field(..., max_length=80)
    display_label: str = Field(..., max_length=120)
    region_grid_rows: int = 3
    region_grid_cols: int = 3


class CharacterCreate(CharacterBase):
    pass


class CharacterOut(CharacterBase):
    id: int

    model_config = ConfigDict(from_attributes=True)

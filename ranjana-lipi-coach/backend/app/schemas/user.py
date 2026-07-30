"""User and auth schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)
    display_name: str = Field(..., max_length=120)


class UserLogin(BaseModel):
    email: str = Field(..., max_length=255)
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

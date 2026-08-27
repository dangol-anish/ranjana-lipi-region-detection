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


class GoogleLogin(BaseModel):
    id_token: str = Field(..., min_length=20)


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    email: str | None = Field(default=None, max_length=255)
    display_name: str | None = Field(default=None, max_length=120)


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

"""initial backend tables

Revision ID: 20260730_0001
Revises:
Create Date: 2026-07-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260730_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "characters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("display_label", sa.String(length=120), nullable=False),
        sa.Column("region_grid_rows", sa.Integer(), nullable=False),
        sa.Column("region_grid_cols", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_characters_id"), "characters", ["id"], unique=False)
    op.create_index(op.f("ix_characters_name"), "characters", ["name"], unique=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)

    op.create_table(
        "attempts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("character_id", sa.Integer(), nullable=False),
        sa.Column(
            "mode",
            sa.Enum(
                "app_suggested",
                "free_practice",
                "assessment",
                name="practice_mode",
            ),
            nullable=False,
        ),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("region_feedback", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_attempts_character_id"), "attempts", ["character_id"], unique=False)
    op.create_index(op.f("ix_attempts_id"), "attempts", ["id"], unique=False)
    op.create_index(op.f("ix_attempts_user_id"), "attempts", ["user_id"], unique=False)

    op.create_table(
        "user_character_progress",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("character_id", sa.Integer(), nullable=False),
        sa.Column("attempts_count", sa.Integer(), nullable=False),
        sa.Column("best_score", sa.Float(), nullable=True),
        sa.Column("last_practiced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mastered", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "character_id", name="uq_user_character_progress"),
    )
    op.create_index(
        op.f("ix_user_character_progress_character_id"),
        "user_character_progress",
        ["character_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_character_progress_id"),
        "user_character_progress",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_character_progress_user_id"),
        "user_character_progress",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_user_character_progress_user_id"), table_name="user_character_progress")
    op.drop_index(op.f("ix_user_character_progress_id"), table_name="user_character_progress")
    op.drop_index(
        op.f("ix_user_character_progress_character_id"),
        table_name="user_character_progress",
    )
    op.drop_table("user_character_progress")
    op.drop_index(op.f("ix_attempts_user_id"), table_name="attempts")
    op.drop_index(op.f("ix_attempts_id"), table_name="attempts")
    op.drop_index(op.f("ix_attempts_character_id"), table_name="attempts")
    op.drop_table("attempts")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_characters_name"), table_name="characters")
    op.drop_index(op.f("ix_characters_id"), table_name="characters")
    op.drop_table("characters")

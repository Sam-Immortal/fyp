"""add nlp event fields

Revision ID: 0002
Revises: 0001
Create Date: 2025-10-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to nlp_score_events table
    # Use ADD COLUMN IF NOT EXISTS for idempotency
    op.execute("""
        ALTER TABLE nlp_score_events 
        ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'pending'
    """)
    op.execute("""
        ALTER TABLE nlp_score_events 
        ADD COLUMN IF NOT EXISTS risk_level VARCHAR(20)
    """)
    op.execute("""
        ALTER TABLE nlp_score_events 
        ADD COLUMN IF NOT EXISTS patterns JSONB
    """)
    op.execute("""
        ALTER TABLE nlp_score_events 
        ADD COLUMN IF NOT EXISTS explanation TEXT
    """)
    op.execute("""
        ALTER TABLE nlp_score_events 
        ADD COLUMN IF NOT EXISTS error TEXT
    """)

    # Update existing rows to have status='completed'
    op.execute("""
        UPDATE nlp_score_events SET status = 'completed' WHERE status IS NULL OR status = 'pending'
    """)


def downgrade() -> None:
    # Remove the added columns
    op.drop_column("nlp_score_events", "error")
    op.drop_column("nlp_score_events", "explanation")
    op.drop_column("nlp_score_events", "patterns")
    op.drop_column("nlp_score_events", "risk_level")
    op.drop_column("nlp_score_events", "status")
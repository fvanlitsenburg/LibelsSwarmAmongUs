"""make relevance sequence provider specific

Revision ID: 9cb04eaae4a0
Revises: 4a6ef28aa57a
Create Date: 2026-08-12 13:00:02.711303

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9cb04eaae4a0'
down_revision: str | Sequence[str] | None = '4a6ef28aa57a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.drop_constraint(
        op.f("uq_relevance_assessment_sequence"),
        "relevance_assessments",
        type_="unique",
    )

    op.create_unique_constraint(
        "uq_relevance_document_provider_sequence",
        "relevance_assessments",
        [
            "document_id",
            "provider",
            "sequence_number",
        ],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_constraint(
        "uq_relevance_document_provider_sequence",
        "relevance_assessments",
        type_="unique",
    )

    op.create_unique_constraint(
        op.f("uq_relevance_assessment_sequence"),
        "relevance_assessments",
        [
            "document_id",
            "sequence_number",
        ],
        postgresql_nulls_not_distinct=False,
    )
"""Shared domain types used by every source and pipeline stage."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Source(StrEnum):
    """Collections currently supported by the pipeline."""

    DUPO = "DUPO"
    TCP = "TCP"


class RelevanceStatus(StrEnum):
    """Current relevance status of a document."""

    NOT_ASSESSED = "not_assessed"
    UNCERTAIN = "uncertain"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"


class ClassificationStatus(StrEnum):
    """How much of the document informed its classification."""

    NOT_CLASSIFIED = "not_classified"
    PARTIAL_TEXT = "partial_text"
    FULL_TEXT = "full_text"


class DocumentMetadata(BaseModel):
    """
    Normalized metadata shared by DUPO and TCP records.

    Source-specific fields remain separate. They should not be combined into
    one ambiguous identifier.
    """

    model_config = ConfigDict(extra="forbid")

    source: Source

    # Stable identifier from the source collection, once known.
    source_record_id: str | None = None

    # Shared bibliographic metadata.
    title: str | None = None
    author: str | None = None
    year: int | None = Field(default=None, ge=1000, le=2100)
    source_date: str | None = None
    language: str | None = None

    # Dutch Pamphlets Online identifiers.
    dupo_id: str | None = None
    knuttel_number: str | None = None

    # TCP metadata and identifiers.
    tcp_id: str | None = None
    eebo_id: str | None = None
    vid: str | None = None
    stc: str | None = None
    source_status: str | None = None
    source_terms: list[str] = Field(default_factory=list)
    source_pages: str | None = None

    # Shared analysis fields.
    relevance_status: RelevanceStatus = RelevanceStatus.NOT_ASSESSED
    primary_category: str | None = None
    topic: str | None = None
    classification_status: ClassificationStatus = (
        ClassificationStatus.NOT_CLASSIFIED
    )
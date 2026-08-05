"""Database models for DUPO and TCP documents."""

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from historical_text_pipeline.db.base import Base
from historical_text_pipeline.domain import (
    ClassificationStatus,
    RelevanceStatus,
    Source,
)


def enum_values(enum_class: type[Enum]) -> list[str]:
    """Store string-enum values rather than Python member names."""

    return [str(member.value) for member in enum_class]


class Document(Base):
    """
    Shared document record.

    This table contains fields used by both DUPO and TCP. Collection-specific
    metadata lives in one-to-one tables.
    """

    __tablename__ = "documents"

    __table_args__ = (
        UniqueConstraint(
            "source",
            "source_record_id",
            name="uq_documents_source_record_id",
        ),
        Index("ix_documents_source_year", "source", "year"),
        Index(
            "ix_documents_relevance_category",
            "relevance_status",
            "primary_category",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    source: Mapped[Source] = mapped_column(
        SqlEnum(
            Source,
            values_callable=enum_values,
            native_enum=False,
            validate_strings=True,
            name="source_enum",
        ),
        nullable=False,
        index=True,
    )

    # Identifier used by the source collection.
    source_record_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Location of the existing source file.
    source_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    source_filename: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )
    file_checksum: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # Shared bibliographic metadata.
    language: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        index=True,
    )
    year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )
    source_date: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    title: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    author: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Processing progress. A unit is a DUPO page or a TCP text chunk.
    total_units: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    units_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    text_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    text_method: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Relevance.
    relevance_status: Mapped[RelevanceStatus] = mapped_column(
        SqlEnum(
            RelevanceStatus,
            values_callable=enum_values,
            native_enum=False,
            validate_strings=True,
            name="relevance_status_enum",
        ),
        nullable=False,
        default=RelevanceStatus.NOT_ASSESSED,
        index=True,
    )
    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    relevance_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    relevance_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Classification.
    primary_category: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    topic: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    classification_status: Mapped[ClassificationStatus] = mapped_column(
        SqlEnum(
            ClassificationStatus,
            values_callable=enum_values,
            native_enum=False,
            validate_strings=True,
            name="classification_status_enum",
        ),
        nullable=False,
        default=ClassificationStatus.NOT_CLASSIFIED,
    )
    classification_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    processing_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="discovered",
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    dupo: Mapped["DupoMetadata | None"] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
    )
    tcp: Mapped["TcpMetadata | None"] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
    )
    text_units: Mapped[list["DocumentTextUnit"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    relevance_assessments: Mapped[list["RelevanceAssessment"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DupoMetadata(Base):
    """Metadata specific to Dutch Pamphlets Online."""

    __tablename__ = "dupo_metadata"

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Identifier used by the online DUPO collection.
    dupo_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
    )

    # Catalogue number printed on or extracted from the first page.
    knuttel_number: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="dupo",
    )


class TcpMetadata(Base):
    """Metadata supplied with TCP raw-text documents."""

    __tablename__ = "tcp_metadata"

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )

    tcp_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
    )
    eebo_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    vid: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    stc: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    source_status: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    source_terms: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    source_pages: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="tcp",
    )


class DocumentTextUnit(Base):
    """
    One processed portion of a document.

    For DUPO, a unit normally represents one OCR page.
    For TCP, a unit normally represents one raw-text chunk.
    """

    __tablename__ = "document_text_units"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "unit_type",
            "unit_number",
            name="uq_document_text_unit",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    unit_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    unit_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    word_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    processing_method: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="text_units",
    )


class RelevanceAssessment(Base):
    """One relevance decision made from the text available at a checkpoint."""

    __tablename__ = "relevance_assessments"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "sequence_number",
            name="uq_relevance_assessment_sequence",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    units_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    decision: Mapped[RelevanceStatus] = mapped_column(
        SqlEnum(
            RelevanceStatus,
            values_callable=enum_values,
            native_enum=False,
            validate_strings=True,
            name="assessment_decision_enum",
        ),
        nullable=False,
    )
    relevance_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    primary_category: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    topic: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    classification_status: Mapped[ClassificationStatus] = mapped_column(
        SqlEnum(
            ClassificationStatus,
            values_callable=enum_values,
            native_enum=False,
            validate_strings=True,
            name="assessment_classification_status_enum",
        ),
        nullable=False,
        default=ClassificationStatus.PARTIAL_TEXT,
    )

    supporting_evidence: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    missing_information: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped[Document] = relationship(
        back_populates="relevance_assessments",
    )
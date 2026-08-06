"""Shared types for relevance assessment."""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class RelevanceDecision(StrEnum):
    """Action recommended after reading the available text."""

    CONTINUE = "continue"
    STOP = "stop"
    UNCERTAIN = "uncertain"


class RelevanceAssessmentOutput(BaseModel):
    """Structured assessment returned by the language model."""

    model_config = ConfigDict(extra="forbid")

    decision: RelevanceDecision

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated probability that the document is relevant."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that the decision is justified by the "
            "available OCR text."
        ),
    )

    reason: str = Field(
        min_length=1,
        max_length=1000,
    )

    category: str = Field(
        min_length=1,
        max_length=100,
    )

    topic: str = Field(
        min_length=1,
        max_length=200,
    )

    supporting_evidence: list[str] = Field(
        default_factory=list,
        max_length=5,
    )

    missing_information: list[str] = Field(
        default_factory=list,
        max_length=5,
    )


class RelevanceAssessor(Protocol):
    """Interface implemented by a relevance-assessment provider."""

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        assessment_number: int,
    ) -> RelevanceAssessmentOutput:
        """Assess the available document text."""

    def close(self) -> None:
        """Release network resources."""
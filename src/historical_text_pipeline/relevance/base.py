"""Shared types for relevance assessment."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from historical_text_pipeline.domain import (
    AnalysisProvider,
)

RELEVANCE_PROMPT_VERSION = "relevance-v1"

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
        max_length=2000,
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
   
@dataclass(frozen=True, slots=True)
class RelevanceAssessmentRun:
    """One provider's progressive relevance call."""

    output: RelevanceAssessmentOutput
    provider: AnalysisProvider
    model: str
    prompt_version: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    

class RelevanceAssessor(Protocol):
    """Interface for progressive relevance assessment."""

    @property
    def provider(self) -> AnalysisProvider:
        """Return the analysis provider."""

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        assessment_number: int,
        final_progressive_assessment: bool = False,
    ) -> RelevanceAssessmentRun:
        """Assess the accumulated document text."""

    def close(self) -> None:
        """Release resources."""
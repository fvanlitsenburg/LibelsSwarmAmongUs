"""Tests for the Anthropic final-assessment adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from historical_text_pipeline.domain import (
    AnalysisProvider,
)
from historical_text_pipeline.relevance.anthropic_final_assessor import (
    AnthropicFinalAssessmentError,
    AnthropicFinalAssessor,
)
from historical_text_pipeline.relevance.final_assessment import (
    FinalAssessmentOutput,
    FinalRelevanceDecision,
)


def make_output() -> FinalAssessmentOutput:
    """Return a valid fake Claude result."""

    return FinalAssessmentOutput(
        decision=FinalRelevanceDecision.RELEVANT,
        relevance_score=0.91,
        confidence=0.88,
        relevance_explanation=(
            "The complete document directly satisfies "
            "the research criteria."
        ),
        category="political polemic",
        topic="Conflict over public authority",
        summary="A concise Claude test summary.",
        supporting_evidence=[
            "Test evidence.",
        ],
        caveats=[],
    )


def make_assessor(
    response: object,
) -> AnthropicFinalAssessor:
    """Build an assessor around a mocked Anthropic client."""

    client = Mock()
    client.messages.parse.return_value = response

    return AnthropicFinalAssessor(
        client=client,
        model="claude-sonnet-5",
        max_output_tokens=4000,
        effort="low",
    )


def test_returns_provider_aware_final_run() -> None:
    response = SimpleNamespace(
        id="msg_test_123",
        model="claude-sonnet-5",
        stop_reason="end_turn",
        parsed_output=make_output(),
        usage=SimpleNamespace(
            input_tokens=1500,
            output_tokens=300,
        ),
    )

    assessor = make_assessor(response)

    result = assessor.assess(
        text="PDF PAGE 1\nComplete OCR text.",
        criteria="Test criteria.",
        document_context="Document ID: 1",
    )

    assert result.provider == (
        AnalysisProvider.ANTHROPIC
    )
    assert result.model == "claude-sonnet-5"
    assert result.response_id == "msg_test_123"
    assert result.input_tokens == 1500
    assert result.output_tokens == 300
    assert result.output.summary == (
        "A concise Claude test summary."
    )


def test_max_tokens_is_rejected() -> None:
    response = SimpleNamespace(
        id="msg_test_123",
        model="claude-sonnet-5",
        stop_reason="max_tokens",
        parsed_output=None,
        usage=SimpleNamespace(
            input_tokens=1500,
            output_tokens=4000,
        ),
    )

    assessor = make_assessor(response)

    with pytest.raises(
        AnthropicFinalAssessmentError,
        match="output-token limit",
    ):
        assessor.assess(
            text="PDF PAGE 1\nComplete OCR text.",
            criteria="Test criteria.",
            document_context="Document ID: 1",
        )
        
    
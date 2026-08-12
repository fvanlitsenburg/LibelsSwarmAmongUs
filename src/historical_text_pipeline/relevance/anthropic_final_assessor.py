"""Final full-text assessment using Anthropic Claude."""

import anthropic
from anthropic import Anthropic

from historical_text_pipeline.config.settings import (
    Settings,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
)
from historical_text_pipeline.relevance.final_assessment import (
    FINAL_ASSESSMENT_INSTRUCTIONS,
    FINAL_PROMPT_VERSION,
    FinalAssessmentOutput,
    FinalAssessmentRun,
)


class AnthropicFinalAssessmentError(Exception):
    """Raised when Claude cannot produce a final assessment."""


class AnthropicFinalAssessor:
    """Assess complete document text using Anthropic Claude."""

    def __init__(
        self,
        *,
        client: Anthropic,
        model: str,
        max_output_tokens: int,
        effort: str,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._effort = effort

    @property
    def provider(self) -> AnalysisProvider:
        """Return the provider represented by this assessor."""

        return AnalysisProvider.ANTHROPIC

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "AnthropicFinalAssessor":
        """Construct the assessor from application settings."""

        if settings.anthropic_api_key is None:
            raise AnthropicFinalAssessmentError(
                "LSAU_ANTHROPIC_API_KEY is not configured."
            )

        client = Anthropic(
            api_key=(
                settings.anthropic_api_key.get_secret_value()
            ),
            timeout=settings.anthropic_timeout_seconds,
            max_retries=2,
        )

        return cls(
            client=client,
            model=settings.anthropic_final_model,
            max_output_tokens=(
                settings.anthropic_final_max_output_tokens
            ),
            effort=settings.anthropic_final_effort,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        document_context: str,
    ) -> FinalAssessmentRun:
        """Assess one complete OCR transcription."""

        if not text.strip():
            raise ValueError(
                "Cannot assess an empty transcription."
            )

        if not criteria.strip():
            raise ValueError(
                "Cannot assess a document without research criteria."
            )

        user_message = f"""
RESEARCH CRITERIA

{criteria}


CATALOGUE CONTEXT

{document_context}


COMPLETE OCR TRANSCRIPTION

{text}
""".strip()

        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_output_tokens,
                system=FINAL_ASSESSMENT_INSTRUCTIONS,
                messages=[
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
                output_format=FinalAssessmentOutput,
                output_config={
                    "effort": self._effort,
                },
            )

        except anthropic.APIError as error:
            raise AnthropicFinalAssessmentError(
                f"Anthropic final assessment failed: {error}"
            ) from error

        if response.stop_reason == "max_tokens":
            raise AnthropicFinalAssessmentError(
                "Claude reached the configured output-token "
                f"limit before completing the assessment. "
                f"Message ID: {response.id}."
            )

        if response.stop_reason == "refusal":
            raise AnthropicFinalAssessmentError(
                "Claude refused the final assessment. "
                f"Message ID: {response.id}."
            )

        if response.stop_reason not in {
            "end_turn",
            "stop_sequence",
        }:
            raise AnthropicFinalAssessmentError(
                "Claude returned an unexpected stop reason: "
                f"{response.stop_reason!r}. "
                f"Message ID: {response.id}."
            )

        output = response.parsed_output

        if output is None:
            raise AnthropicFinalAssessmentError(
                "Claude returned no parsed final assessment. "
                f"Message ID: {response.id}."
            )

        return FinalAssessmentRun(
            output=output,
            provider=AnalysisProvider.ANTHROPIC,
            model=response.model,
            prompt_version=FINAL_PROMPT_VERSION,
            response_id=response.id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
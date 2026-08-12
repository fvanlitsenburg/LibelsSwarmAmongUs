"""Progressive relevance assessment using Anthropic Claude."""

import anthropic
from anthropic import Anthropic

from historical_text_pipeline.config.settings import (
    Settings,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
)
from historical_text_pipeline.relevance.base import (
    RELEVANCE_PROMPT_VERSION,
    RelevanceAssessmentOutput,
    RelevanceAssessmentRun,
)
from historical_text_pipeline.relevance.openai_assessor import (
    RELEVANCE_INSTRUCTIONS,
)


class AnthropicRelevanceError(Exception):
    """Raised when Claude relevance assessment fails."""


class AnthropicRelevanceAssessor:
    """Assess progressive OCR text using Claude."""

    def __init__(
        self,
        *,
        client: Anthropic,
        model: str,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    @property
    def provider(self) -> AnalysisProvider:
        """Return the provider represented by this assessor."""

        return AnalysisProvider.ANTHROPIC

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "AnthropicRelevanceAssessor":
        """Construct the assessor from application settings."""

        if settings.anthropic_api_key is None:
            raise AnthropicRelevanceError(
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
            model=settings.anthropic_relevance_model,
            max_output_tokens=(
                settings.anthropic_relevance_max_output_tokens
            ),
        )

    def close(self) -> None:
        """Close the Anthropic client."""

        self._client.close()

    def assess(
        self,
        *,
        text: str,
        criteria: str,
        assessment_number: int,
        final_progressive_assessment: bool = False,
    ) -> RelevanceAssessmentRun:
        """Assess accumulated OCR text."""

        if final_progressive_assessment:
            assessment_mode = """
This is the FINAL progressive relevance assessment.

Choose CONTINUE only if the accumulated text contains direct,
substantive evidence that the document satisfies the research
criteria.

Do not return UNCERTAIN merely because later pages might contain
relevant material.

Return UNCERTAIN only when OCR quality is genuinely too poor to
make a defensible decision.
""".strip()

        else:
            assessment_mode = """
This is an intermediate progressive relevance assessment.

Choose UNCERTAIN only when additional pages are genuinely needed
to determine whether the document satisfies the research criteria.
""".strip()

        user_message = f"""
RESEARCH CRITERIA

{criteria}


ASSESSMENT NUMBER

{assessment_number}


ASSESSMENT MODE

{assessment_mode}


ACCUMULATED OCR

{text}
""".strip()

        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_output_tokens,
                system=RELEVANCE_INSTRUCTIONS,
                messages=[
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
                output_format=RelevanceAssessmentOutput,
            )

        except anthropic.APIError as error:
            raise AnthropicRelevanceError(
                f"Anthropic relevance request failed: {error}"
            ) from error

        if response.stop_reason == "max_tokens":
            raise AnthropicRelevanceError(
                "Claude reached the configured output-token "
                f"limit. Message ID: {response.id}."
            )

        if response.stop_reason == "refusal":
            raise AnthropicRelevanceError(
                "Claude refused the relevance assessment. "
                f"Message ID: {response.id}."
            )

        output = response.parsed_output

        if output is None:
            raise AnthropicRelevanceError(
                "Claude returned no parsed relevance assessment. "
                f"Message ID: {response.id}."
            )

        return RelevanceAssessmentRun(
            output=output,
            provider=AnalysisProvider.ANTHROPIC,
            model=response.model,
            prompt_version=RELEVANCE_PROMPT_VERSION,
            response_id=response.id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
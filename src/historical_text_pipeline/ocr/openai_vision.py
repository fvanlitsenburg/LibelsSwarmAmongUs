"""OpenAI vision implementation of the OCR backend."""

from base64 import b64encode

import openai
from openai import OpenAI

from historical_text_pipeline.config.settings import Settings
from historical_text_pipeline.ocr.base import OcrPageResult

DIPLOMATIC_TRANSCRIPTION_INSTRUCTIONS = """
You are transcribing one page of an early modern printed historical document.

Produce a diplomatic transcription of only the text visibly present in the
image.

Rules:
- Preserve original spelling.
- Preserve capitalization and punctuation.
- Preserve paragraph divisions and line breaks as closely as possible.
- Preserve printed hyphenation at line endings.
- Include titles, running headers, page numbers, signatures, and catchwords.
- Preserve the long s as ſ when it is visually distinguishable.
- Do not translate the text.
- Do not modernize spelling.
- Do not silently correct printing errors.
- Do not summarize or explain the document.
- Do not complete missing text from context.
- Mark an uncertain reading as ⟦reading?⟧.
- Mark unreadable text as ⟦illegible⟧.
- Return only the transcription.
- Do not use Markdown or a code block.
""".strip()


class OpenAiOcrError(Exception):
    """Raised when OpenAI cannot transcribe an image."""


class OpenAiConfigurationError(OpenAiOcrError):
    """Raised when required OpenAI configuration is absent."""


class OpenAiOcrBackend:
    """Recognize historical document images with OpenAI vision."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "OpenAiOcrBackend":
        """Construct the backend from application settings."""

        if settings.openai_api_key is None:
            raise OpenAiConfigurationError(
                "LSAU_OPENAI_API_KEY is not configured."
            )

        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=2,
        )

        return cls(
            client=client,
            model=settings.openai_ocr_model,
            max_output_tokens=settings.openai_ocr_max_output_tokens,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
    ) -> OcrPageResult:
        """Transcribe one historical document image."""

        if not image_bytes:
            raise ValueError("The image is empty.")

        if not mime_type.startswith("image/"):
            raise ValueError(
                f"Expected an image MIME type, received: {mime_type}"
            )

        encoded_image = b64encode(image_bytes).decode("ascii")
        image_data_url = (
            f"data:{mime_type};base64,{encoded_image}"
        )

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=DIPLOMATIC_TRANSCRIPTION_INSTRUCTIONS,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Transcribe this page according to "
                                    "the supplied rules."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": image_data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
                max_output_tokens=self._max_output_tokens,
                store=False,
            )

        except openai.APIStatusError as error:
            request_id = (
                f" Request ID: {error.request_id}."
                if error.request_id
                else ""
            )

            raise OpenAiOcrError(
                f"OpenAI returned HTTP {error.status_code}."
                f"{request_id} {error}"
            ) from error

        except openai.OpenAIError as error:
            raise OpenAiOcrError(
                f"The OpenAI transcription request failed: {error}"
            ) from error

        transcription = response.output_text.strip()

        if not transcription:
            raise OpenAiOcrError(
                "OpenAI returned no transcription text."
            )

        return OcrPageResult(
            provider=self.provider_name,
            model=str(response.model),
            response_id=response.id,
            text=transcription,
        )
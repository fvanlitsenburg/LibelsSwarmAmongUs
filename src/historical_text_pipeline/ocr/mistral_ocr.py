"""Mistral Document AI implementation of the OCR backend."""

from base64 import b64encode
from typing import Any

import httpx

from historical_text_pipeline.config.settings import Settings
from historical_text_pipeline.ocr.base import OcrPageResult


class MistralOcrError(Exception):
    """Raised when Mistral cannot process an image."""


class MistralConfigurationError(MistralOcrError):
    """Raised when Mistral configuration is missing."""


class MistralOcrBackend:
    """Recognize document images with Mistral Document AI."""

    provider_name = "mistral"

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str,
        model: str,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "MistralOcrBackend":
        """Construct the backend from application settings."""

        if settings.mistral_api_key is None:
            raise MistralConfigurationError(
                "LSAU_MISTRAL_API_KEY is not configured."
            )

        base_url = settings.mistral_api_base_url.rstrip("/") + "/"

        client = httpx.Client(
            base_url=base_url,
            timeout=settings.mistral_timeout_seconds,
            follow_redirects=True,
        )

        return cls(
            client=client,
            api_key=settings.mistral_api_key.get_secret_value(),
            model=settings.mistral_ocr_model,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""

        self._client.close()

    def recognize_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
    ) -> OcrPageResult:
        """Transcribe one page image with Mistral OCR."""

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
            response = self._client.post(
                "ocr",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "document": {
                        "type": "image_url",
                        "image_url": image_data_url,
                    },
                    # We only need transcription text at this stage.
                    "include_image_base64": False,
                    "include_blocks": False,
                },
            )

        except httpx.TimeoutException as error:
            raise MistralOcrError(
                "The Mistral OCR request timed out."
            ) from error

        except httpx.RequestError as error:
            raise MistralOcrError(
                f"Could not connect to Mistral OCR: {error}"
            ) from error

        payload = self._read_response(response)
        transcription = self._extract_text(payload)

        response_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-mistral-request-id")
        )

        model = payload.get("model", self._model)

        return OcrPageResult(
            provider=self.provider_name,
            model=str(model),
            text=transcription,
            response_id=response_id,
        )

    @staticmethod
    def _read_response(
        response: httpx.Response,
    ) -> dict[str, Any]:
        """Read a successful response or expose the API error."""

        try:
            payload = response.json()
        except ValueError as error:
            raise MistralOcrError(
                f"Mistral returned HTTP {response.status_code} "
                "with a non-JSON response."
            ) from error

        if response.is_error:
            if isinstance(payload, dict):
                details = (
                    payload.get("message")
                    or payload.get("detail")
                    or payload
                )
            else:
                details = payload

            raise MistralOcrError(
                f"Mistral returned HTTP {response.status_code}: "
                f"{details}"
            )

        if not isinstance(payload, dict):
            raise MistralOcrError(
                "Mistral returned an unexpected response."
            )

        return payload

    @staticmethod
    def _extract_text(
        payload: dict[str, Any],
    ) -> str:
        """Combine the Markdown returned for each page."""

        pages = payload.get("pages")

        if not isinstance(pages, list) or not pages:
            raise MistralOcrError(
                "Mistral returned no OCR pages."
            )

        page_texts: list[str] = []

        for page in pages:
            if not isinstance(page, dict):
                continue

            markdown = page.get("markdown")

            if isinstance(markdown, str) and markdown.strip():
                page_texts.append(markdown.strip())

        transcription = "\n\n".join(page_texts)

        if not transcription:
            raise MistralOcrError(
                "Mistral returned no transcription text."
            )

        return transcription
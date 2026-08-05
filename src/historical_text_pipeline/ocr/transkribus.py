"""Transkribus implementation of the OCR backend."""

from base64 import b64encode
from time import monotonic, sleep
from typing import Any

import httpx

from historical_text_pipeline.config.settings import Settings
from historical_text_pipeline.ocr.base import OcrPageResult
from historical_text_pipeline.ocr.transkribus_auth import (
    TranskribusAuthenticator,
)

TERMINAL_STATUSES = {"FINISHED", "FAILED"}


class TranskribusError(Exception):
    """Base exception for Transkribus processing failures."""


class TranskribusApiError(TranskribusError):
    """Raised when the processing API returns an error response."""


class TranskribusProcessingFailedError(TranskribusError):
    """Raised when a submitted Transkribus job fails."""


class TranskribusTimeoutError(TranskribusError):
    """Raised when a Transkribus job exceeds the configured timeout."""


class TranskribusOcrBackend:
    """Recognize page images through the Transkribus processing API."""

    provider_name = "transkribus"

    def __init__(
        self,
        *,
        client: httpx.Client,
        authenticator: TranskribusAuthenticator,
        api_base_url: str,
        model_id: int,
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self._client = client
        self._authenticator = authenticator
        self._api_base_url = api_base_url.rstrip("/")
        self._model_id = model_id
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "TranskribusOcrBackend":
        """Construct the backend from application settings."""

        client = httpx.Client(
            timeout=30,
            follow_redirects=True,
        )

        authenticator = TranskribusAuthenticator(
            client=client,
            token_url=settings.transkribus_token_url,
            client_id=settings.transkribus_client_id,
            username=settings.transkribus_username,
            password=settings.transkribus_password,
        )

        return cls(
            client=client,
            authenticator=authenticator,
            api_base_url=settings.transkribus_api_base_url,
            model_id=settings.transkribus_model_id,
            poll_interval_seconds=(
                settings.transkribus_poll_interval_seconds
            ),
            timeout_seconds=settings.transkribus_timeout_seconds,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def recognize_image(
        self,
        image_bytes: bytes,
    ) -> OcrPageResult:
        """Submit one image and wait for its transcription."""

        process_id = self._submit_image(image_bytes)
        payload = self._wait_for_result(process_id)

        content = payload.get("content")

        if not isinstance(content, dict):
            raise TranskribusApiError(
                "Finished Transkribus response contained no content."
            )

        text = content.get("text")

        if not isinstance(text, str):
            raise TranskribusApiError(
                "Finished Transkribus response contained no text."
            )

        return OcrPageResult(
            provider=self.provider_name,
            model_id=str(self._model_id),
            process_id=str(process_id),
            text=text,
        )

    def _authorization_headers(self) -> dict[str, str]:
        token = self._authenticator.get_access_token()

        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _submit_image(self, image_bytes: bytes) -> int:
        print(self._authorization_headers())
        encoded_image = b64encode(image_bytes).decode("ascii")

        response = self._client.post(
            f"{self._api_base_url}/processes",
            headers=self._authorization_headers(),
            json={
                "config": {
                    "textRecognition": {
                        "htrId": self._model_id
                    }
                },
                "image": {
                    "base64": encoded_image,
                },
            },
        )

        payload = self._read_json_response(response)

        try:
            return int(payload["processId"])
        except (KeyError, TypeError, ValueError) as error:
            raise TranskribusApiError(
                "Transkribus did not return a process ID."
            ) from error

    def _wait_for_result(
        self,
        process_id: int,
    ) -> dict[str, Any]:
        deadline = monotonic() + self._timeout_seconds

        while True:
            response = self._client.get(
                f"{self._api_base_url}/processes/{process_id}",
                headers=self._authorization_headers(),
            )

            payload = self._read_json_response(response)
            status = str(payload.get("status", ""))

            if status == "FINISHED":
                return payload

            if status == "FAILED":
                raise TranskribusProcessingFailedError(
                    f"Transkribus process {process_id} failed."
                )

            if monotonic() >= deadline:
                raise TranskribusTimeoutError(
                    f"Transkribus process {process_id} did not "
                    f"finish within {self._timeout_seconds} seconds."
                )

            sleep(self._poll_interval_seconds)

    @staticmethod
    def _read_json_response(
        response: httpx.Response,
    ) -> dict[str, Any]:
        """
        Read a JSON response and preserve useful API error details.

        Transkribus normally returns structured JSON error responses. Include
        the HTTP status and message so configuration and account problems can
        be diagnosed.
        """

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.is_error:
            if isinstance(payload, dict):
                details = (
                    payload.get("message")
                    or payload.get("reasonPhrase")
                    or str(payload)
                )
            else:
                details = response.text.strip()[:2000]

                if not details:
                    details = "The server returned an empty response body."

            raise TranskribusApiError(
                f"{response.request.method} {response.request.url} returned "
                f"{response.status_code} {response.reason_phrase}: {details}"
            )

        if not isinstance(payload, dict):
            raise TranskribusApiError(
                f"{response.request.method} {response.request.url} returned "
                "an unexpected non-JSON response."
            )

        return payload
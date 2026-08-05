"""Tests for the Transkribus OCR backend."""

import json

import httpx
from pydantic import SecretStr

from historical_text_pipeline.ocr.transkribus import (
    TranskribusOcrBackend,
)
from historical_text_pipeline.ocr.transkribus_auth import (
    TranskribusAuthenticator,
)


def test_transkribus_recognizes_one_image() -> None:
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)

        if request.url.path.endswith("/token"):
            return httpx.Response(
                200,
                json={
                    "access_token": "test-token",
                    "refresh_token": "test-refresh-token",
                    "expires_in": 300,
                },
            )

        if (
            request.method == "POST"
            and request.url.path.endswith("/processes")
        ):
            payload = json.loads(request.content)

            assert payload["config"]["textRecognition"] == {
                "htrId": 12345,
                "languageModel": "built-in",
            }
            assert "base64" in payload["image"]

            return httpx.Response(
                200,
                json={
                    "processId": 987,
                    "status": "CREATED",
                },
            )

        if (
            request.method == "GET"
            and request.url.path.endswith("/processes/987")
        ):
            return httpx.Response(
                200,
                json={
                    "processId": 987,
                    "status": "FINISHED",
                    "content": {
                        "text": "Recognized historical text.",
                    },
                },
            )

        raise AssertionError(f"Unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    authenticator = TranskribusAuthenticator(
        client=client,
        token_url="https://example.test/token",
        client_id="processing-api-client",
        username="test-user",
        password=SecretStr("test-password"),
    )

    backend = TranskribusOcrBackend(
        client=client,
        authenticator=authenticator,
        api_base_url="https://example.test/processing/v1",
        model_id=12345,
        poll_interval_seconds=0,
        timeout_seconds=10,
    )

    try:
        result = backend.recognize_image(b"fake JPEG bytes")
    finally:
        backend.close()

    assert result.provider == "transkribus"
    assert result.model_id == "12345"
    assert result.process_id == "987"
    assert result.text == "Recognized historical text."

    assert len(requests_seen) == 3
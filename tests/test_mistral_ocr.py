"""Tests for the Mistral OCR backend."""

import json

import httpx

from historical_text_pipeline.ocr.mistral_ocr import (
    MistralOcrBackend,
)


def test_mistral_backend_transcribes_image() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/ocr"

        assert request.headers["Authorization"] == (
            "Bearer test-api-key"
        )

        payload = json.loads(request.content)

        assert payload["model"] == "mistral-ocr-4-0"
        assert payload["document"]["type"] == "image_url"
        assert payload["document"]["image_url"] == (
            "data:image/jpeg;base64,ZmFrZSBqcGVn"
        )
        assert payload["include_blocks"] is False

        return httpx.Response(
            200,
            headers={
                "x-request-id": "req_mistral_123",
            },
            json={
                "model": "mistral-ocr-4-0",
                "pages": [
                    {
                        "index": 0,
                        "markdown": (
                            "Dit is de getranscribeerde tekst."
                        ),
                    }
                ],
                "usage_info": {
                    "pages_processed": 1,
                },
            },
        )

    client = httpx.Client(
        base_url="https://api.mistral.ai/v1/",
        transport=httpx.MockTransport(handler),
    )

    backend = MistralOcrBackend(
        client=client,
        api_key="test-api-key",
        model="mistral-ocr-4-0",
    )

    try:
        result = backend.recognize_image(
            b"fake jpeg",
            mime_type="image/jpeg",
        )
    finally:
        backend.close()

    assert result.provider == "mistral"
    assert result.model == "mistral-ocr-4-0"
    assert result.response_id == "req_mistral_123"
    assert result.text == (
        "Dit is de getranscribeerde tekst."
    )
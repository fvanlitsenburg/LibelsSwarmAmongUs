"""Tests for the OpenAI vision OCR backend."""

from typing import cast

from openai import OpenAI

from historical_text_pipeline.ocr.openai_vision import (
    OpenAiOcrBackend,
)


class FakeResponse:
    """A minimal fake OpenAI response."""

    id = "resp_test_123"
    model = "gpt-5.5"
    output_text = "Dit is de getranscribeerde tekst."


class FakeResponses:
    """Record the request made to responses.create."""

    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> FakeResponse:
        self.request = kwargs
        return FakeResponse()


class FakeOpenAiClient:
    """A minimal fake OpenAI client."""

    def __init__(self) -> None:
        self.responses = FakeResponses()

    def close(self) -> None:
        """Match the real client's close method."""


def test_openai_backend_transcribes_image() -> None:
    fake_client = FakeOpenAiClient()

    backend = OpenAiOcrBackend(
        client=cast(OpenAI, fake_client),
        model="gpt-5.5",
        max_output_tokens=12_000,
    )

    result = backend.recognize_image(
        b"fake jpeg",
        mime_type="image/jpeg",
    )

    assert result.provider == "openai"
    assert result.model == "gpt-5.5"
    assert result.response_id == "resp_test_123"
    assert result.text == "Dit is de getranscribeerde tekst."

    request = fake_client.responses.request

    assert request is not None
    assert request["model"] == "gpt-5.5"
    assert request["max_output_tokens"] == 12_000
    assert request["store"] is False

    input_items = request["input"]
    assert isinstance(input_items, list)

    message = input_items[0]
    assert isinstance(message, dict)

    content = message["content"]
    assert isinstance(content, list)

    image_input = content[1]
    assert image_input["type"] == "input_image"
    assert image_input["detail"] == "high"
    assert image_input["image_url"] == (
        "data:image/jpeg;base64,ZmFrZSBqcGVn"
    )
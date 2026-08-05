"""Tests for PDF inspection."""

from pathlib import Path

import pytest

from historical_text_pipeline.ingest import pdf_inspection
from historical_text_pipeline.ingest.pdf_inspection import (
    calculate_alphabetic_ratio,
    inspect_pdf,
)


class FakePage:
    """A fake PDF page with predefined extracted text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakeReader:
    """A fake unencrypted PDF reader."""

    pages: list[FakePage]
    is_encrypted = False

    def __init__(
        self,
        path: Path,
        strict: bool = False,
    ) -> None:
        del path, strict


def test_calculate_alphabetic_ratio() -> None:
    assert calculate_alphabetic_ratio("") == 0.0
    assert calculate_alphabetic_ratio("abc") == 1.0
    assert calculate_alphabetic_ratio("abc 123") == pytest.approx(0.5)


def test_pdf_with_substantial_text_is_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    text = (
        "This is a substantial amount of readable historical text. "
        * 20
    )

    FakeReader.pages = [
        FakePage(text),
        FakePage(text),
        FakePage(text),
    ]

    monkeypatch.setattr(
        pdf_inspection,
        "PdfReader",
        FakeReader,
    )

    result = inspect_pdf(pdf_path)

    assert result.page_count == 3
    assert result.sample_pages == 3
    assert result.extracted_characters >= 200
    assert result.usable_text_layer is True


def test_image_only_pdf_requires_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"fake PDF")

    FakeReader.pages = [
        FakePage(""),
        FakePage(""),
        FakePage(""),
    ]

    monkeypatch.setattr(
        pdf_inspection,
        "PdfReader",
        FakeReader,
    )

    result = inspect_pdf(pdf_path)

    assert result.page_count == 3
    assert result.extracted_characters == 0
    assert result.usable_text_layer is False
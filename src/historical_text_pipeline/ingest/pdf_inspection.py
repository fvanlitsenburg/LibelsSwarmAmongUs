"""Inspect PDFs before deciding whether OCR is required."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

DEFAULT_SAMPLE_PAGE_LIMIT = 3
DEFAULT_MINIMUM_CHARACTERS = 200
DEFAULT_MINIMUM_ALPHABETIC_RATIO = 0.40


class PdfInspectionError(Exception):
    """Raised when a PDF cannot be inspected."""


class EncryptedPdfError(PdfInspectionError):
    """Raised when a PDF requires an unavailable password."""


@dataclass(frozen=True, slots=True)
class PdfInspection:
    """Information discovered by inspecting one PDF."""

    path: Path
    page_count: int
    sample_pages: int
    extracted_characters: int
    alphabetic_ratio: float
    usable_text_layer: bool


def calculate_alphabetic_ratio(text: str) -> float:
    """
    Calculate the share of non-whitespace characters that are letters.

    This is a basic safeguard against treating page numbers, punctuation,
    or corrupted extraction output as usable prose.
    """

    visible_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    if not visible_characters:
        return 0.0

    alphabetic_characters = sum(
        character.isalpha()
        for character in visible_characters
    )

    return alphabetic_characters / len(visible_characters)


def inspect_pdf(
    path: Path,
    *,
    sample_page_limit: int = DEFAULT_SAMPLE_PAGE_LIMIT,
    minimum_characters: int = DEFAULT_MINIMUM_CHARACTERS,
    minimum_alphabetic_ratio: float = DEFAULT_MINIMUM_ALPHABETIC_RATIO,
) -> PdfInspection:
    """
    Count PDF pages and test whether its existing text layer is usable.

    Only the first few pages are sampled. No OCR is performed.
    """

    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF does not exist: {path}")

    if not path.is_file():
        raise PdfInspectionError(f"PDF path is not a file: {path}")

    try:
        reader = PdfReader(path, strict=False)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as error:
                raise EncryptedPdfError(
                    f"PDF requires a password: {path}"
                ) from error

        page_count = len(reader.pages)

        if page_count == 0:
            raise PdfInspectionError(f"PDF has no pages: {path}")

        sample_pages = min(page_count, sample_page_limit)
        extracted_parts: list[str] = []

        for page_number in range(sample_pages):
            extracted_text = reader.pages[page_number].extract_text() or ""
            extracted_parts.append(extracted_text)

    except FileNotDecryptedError as error:
        raise EncryptedPdfError(
            f"PDF requires a password: {path}"
        ) from error

    except PdfReadError as error:
        raise PdfInspectionError(
            f"PDF could not be read: {path}"
        ) from error

    combined_text = "\n".join(extracted_parts)
    normalized_text = "".join(
        character
        for character in combined_text
        if not character.isspace()
    )

    extracted_characters = len(normalized_text)
    alphabetic_ratio = calculate_alphabetic_ratio(combined_text)

    usable_text_layer = (
        extracted_characters >= minimum_characters
        and alphabetic_ratio >= minimum_alphabetic_ratio
    )

    return PdfInspection(
        path=path,
        page_count=page_count,
        sample_pages=sample_pages,
        extracted_characters=extracted_characters,
        alphabetic_ratio=alphabetic_ratio,
        usable_text_layer=usable_text_layer,
    )
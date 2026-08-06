"""Extract Knuttel numbers from the left half of DUPO first pages."""

import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DupoMetadata,
)
from historical_text_pipeline.domain import Source
from historical_text_pipeline.ocr.base import OcrBackend
from historical_text_pipeline.ocr.pdf_rendering import (
    render_first_pdf_page_knuttel_region_as_jpeg,
)

KNUTTEL_LABEL_PATTERN = re.compile(
    r"""
    \bknuttel
    (?:\s*(?:no|nr)\.?)?
    \s*[:#.-]?\s*
    (?P<number>\d{1,6})
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

EDGE_NUMBER_PATTERN = re.compile(
    r"""
    ^[\s*_`#-]*
    (?:
        (?:knuttel\s*)?
        (?:no|nr)\.?
        \s*[:#.-]?\s*
    )?
    (?P<number>\d{1,6})
    [\s*_`#.,;-]*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class KnuttelExtraction:
    """Result of OCR and Knuttel-number extraction."""

    knuttel_number: str | None
    candidates: tuple[str, ...]
    ocr_text: str
    provider: str
    model: str
    response_id: str | None
    used_embedded_image_fallback: bool = False
    embedded_images_checked: int = 0

class DupoKnuttelError(Exception):
    """Raised when Knuttel metadata cannot be processed."""


@dataclass(frozen=True, slots=True)
class KnuttelSaveResult:
    """Result of attempting to save a Knuttel number."""

    document_id: int
    knuttel_number: str | None
    candidates: tuple[str, ...]
    saved: bool
    skipped_existing: bool
    used_embedded_image_fallback: bool

def _unique_in_order(values: list[str]) -> tuple[str, ...]:
    """Remove duplicate strings while preserving their order."""

    return tuple(dict.fromkeys(values))


def find_knuttel_candidates(text: str) -> tuple[str, ...]:
    """
    Find plausible Knuttel numbers in cropped OCR text.

    Explicit references to 'Knuttel' can appear anywhere. Otherwise,
    standalone numbers are considered only near the beginning or end,
    where stamps and catalogue marks commonly occur in OCR reading order.
    """

    labelled_matches = [
        match.group("number")
        for match in KNUTTEL_LABEL_PATTERN.finditer(text)
    ]

    labelled_candidates = _unique_in_order(labelled_matches)

    if labelled_candidates:
        return labelled_candidates

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    edge_lines = lines[:12] + lines[-12:]

    edge_matches: list[str] = []

    for line in edge_lines:
        match = EDGE_NUMBER_PATTERN.fullmatch(line)

        if match is not None:
            edge_matches.append(match.group("number"))

    return _unique_in_order(edge_matches)


def find_knuttel_number(text: str) -> str | None:
    """
    Return a Knuttel number only when the result is unambiguous.

    Ambiguous OCR should be reviewed rather than silently choosing one
    number.
    """

    candidates = find_knuttel_candidates(text)

    if len(candidates) == 1:
        return candidates[0]

    return None


def extract_knuttel_number_from_first_page(
    pdf_path: Path,
    *,
    backend: OcrBackend,
    dpi: int = 300,
    jpeg_quality: int = 95,
) -> KnuttelExtraction:
    """
    OCR the Knuttel region and extract its catalogue number.

    When Mistral classifies the handwritten number as an image instead
    of text, each extracted image is OCR'd once as a fallback.
    """

    rendered_page = (
        render_first_pdf_page_knuttel_region_as_jpeg(
            pdf_path,
            dpi=dpi,
            jpeg_quality=jpeg_quality,
        )
    )

    primary_result = backend.recognize_image(
        rendered_page.image_bytes,
        mime_type=rendered_page.mime_type,
        include_embedded_images=True,
    )

    primary_candidates = find_knuttel_candidates(
        primary_result.text
    )

    fallback_candidates: list[str] = []
    fallback_texts: list[str] = []
    embedded_images_checked = 0

    if len(primary_candidates) != 1:
        for embedded_image in primary_result.embedded_images:
            embedded_images_checked += 1

            fallback_result = backend.recognize_image(
                embedded_image.image_bytes,
                mime_type=embedded_image.mime_type,
                include_embedded_images=False,
            )

            fallback_texts.append(
                f"[Embedded image: {embedded_image.image_id}]\n"
                f"{fallback_result.text}"
            )

            fallback_candidates.extend(
                find_knuttel_candidates(
                    fallback_result.text
                )
            )

    unique_fallback_candidates = _unique_in_order(
        fallback_candidates
    )

    # A unique result from the isolated embedded image is stronger than
    # ambiguous text from the larger crop.
    if len(unique_fallback_candidates) == 1:
        candidates = unique_fallback_candidates
        knuttel_number = unique_fallback_candidates[0]

    elif len(primary_candidates) == 1:
        candidates = primary_candidates
        knuttel_number = primary_candidates[0]

    else:
        candidates = _unique_in_order(
            [
                *primary_candidates,
                *unique_fallback_candidates,
            ]
        )

        knuttel_number = (
            candidates[0]
            if len(candidates) == 1
            else None
        )

    combined_text_parts = [primary_result.text]

    if fallback_texts:
        combined_text_parts.append(
            "\n\n".join(fallback_texts)
        )

    return KnuttelExtraction(
        knuttel_number=knuttel_number,
        candidates=candidates,
        ocr_text="\n\n".join(combined_text_parts),
        provider=primary_result.provider,
        model=primary_result.model,
        response_id=primary_result.response_id,
        used_embedded_image_fallback=(
            embedded_images_checked > 0
        ),
        embedded_images_checked=embedded_images_checked,
    )
    
def extract_and_save_knuttel_number(
        session: Session,
        *,
        document_id: int,
        backend: OcrBackend,
        dpi: int = 300,
        jpeg_quality: int = 95,
        overwrite: bool = False,
    ) -> KnuttelSaveResult:
        """
        Extract and save the Knuttel number for one DUPO document.

        Existing metadata is not overwritten unless overwrite=True.
        An ambiguous extraction is never saved.
        """

        document = session.get(Document, document_id)

        if document is None:
            raise DupoKnuttelError(
                f"Document {document_id} does not exist."
            )

        if document.source != Source.DUPO:
            raise DupoKnuttelError(
                f"Document {document_id} is not a DUPO document."
            )

        if document.source_path is None:
            raise DupoKnuttelError(
                f"Document {document_id} has no source path."
            )

        if document.dupo is None:
            document.dupo = DupoMetadata()

        existing_number = document.dupo.knuttel_number

        if existing_number and not overwrite:
            return KnuttelSaveResult(
                document_id=document.id,
                knuttel_number=existing_number,
                candidates=(existing_number,),
                saved=False,
                skipped_existing=True,
                used_embedded_image_fallback=False,
            )

        extraction = extract_knuttel_number_from_first_page(
            Path(document.source_path),
            backend=backend,
            dpi=dpi,
            jpeg_quality=jpeg_quality,
        )

        # Do not replace or clear metadata unless one number was identified.
        if extraction.knuttel_number is None:
            return KnuttelSaveResult(
                document_id=document.id,
                knuttel_number=None,
                candidates=extraction.candidates,
                saved=False,
                skipped_existing=False,
                used_embedded_image_fallback=(
                    extraction.used_embedded_image_fallback
                ),
            )

        document.dupo.knuttel_number = extraction.knuttel_number
        session.flush()

        return KnuttelSaveResult(
            document_id=document.id,
            knuttel_number=extraction.knuttel_number,
            candidates=extraction.candidates,
            saved=True,
            skipped_existing=False,
            used_embedded_image_fallback=(
                extraction.used_embedded_image_fallback
            ),
        )
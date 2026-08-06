"""Tests for PDF image preparation."""

from PIL import Image

from historical_text_pipeline.ocr.pdf_rendering import (
    crop_knuttel_region,
)


def test_knuttel_crop_removes_ruler_and_keeps_full_height() -> None:
    image = Image.new(
        "RGB",
        size=(1000, 600),
    )

    try:
        cropped = crop_knuttel_region(image)

        try:
            # Five percent is removed from the left.
            # The crop stops at 49 percent of the full width.
            assert cropped.size == (440, 600)
        finally:
            cropped.close()

    finally:
        image.close()


def test_knuttel_crop_fractions_must_be_valid() -> None:
    image = Image.new(
        "RGB",
        size=(1000, 600),
    )

    try:
        try:
            crop_knuttel_region(
                image,
                left_fraction=0.50,
                right_fraction=0.40,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Invalid crop fractions should raise ValueError."
            )
    finally:
        image.close()
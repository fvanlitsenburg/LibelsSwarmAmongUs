"""Tests for finding DUPO PDFs."""

from pathlib import Path

import pytest

from historical_text_pipeline.ingest.dupo import find_dupo_pdfs


def create_file(path: Path, contents: bytes = b"test") -> None:
    """Create a test file and its parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def test_find_dupo_pdfs(tmp_path: Path) -> None:
    create_file(tmp_path / "1650" / "first.pdf")
    create_file(tmp_path / "1650" / "second.PDF")
    create_file(tmp_path / "1651" / "third.pdf")

    # Ignored because it is not a PDF.
    create_file(tmp_path / "1650" / "notes.txt")

    # Ignored because the directory is not a four-digit year.
    create_file(tmp_path / "misc" / "ignored.pdf")

    # Ignored because nested directories are not supported.
    create_file(tmp_path / "1651" / "nested" / "ignored.pdf")

    documents = find_dupo_pdfs(tmp_path)

    assert len(documents) == 3

    assert documents[0].year == 1650
    assert documents[0].path.name == "first.pdf"

    assert documents[1].year == 1650
    assert documents[1].path.name == "second.PDF"

    assert documents[2].year == 1651
    assert documents[2].path.name == "third.pdf"


def test_missing_dupo_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        find_dupo_pdfs(missing)


def test_dupo_path_must_be_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")

    with pytest.raises(NotADirectoryError):
        find_dupo_pdfs(file_path)
"""Find DUPO PDFs inside configured year directories."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DupoPdf:
    """One DUPO PDF found on disk."""

    path: Path
    year: int


def find_dupo_pdfs(root: Path) -> list[DupoPdf]:
    """
    Find PDFs directly inside four-digit year folders.

    Expected structure:

        root/
        ├── 1650/
        │   ├── file1.pdf
        │   └── file2.pdf
        └── 1651/
            └── file3.pdf
    """

    root = root.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"DUPO directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"DUPO path is not a directory: {root}")

    documents: list[DupoPdf] = []

    for year_directory in sorted(root.iterdir()):
        if not year_directory.is_dir():
            continue

        if len(year_directory.name) != 4:
            continue

        if not year_directory.name.isdigit():
            continue

        year = int(year_directory.name)

        for candidate in sorted(year_directory.iterdir()):
            if not candidate.is_file():
                continue

            if candidate.suffix.casefold() != ".pdf":
                continue

            documents.append(
                DupoPdf(
                    path=candidate.resolve(),
                    year=year,
                )
            )

    return documents
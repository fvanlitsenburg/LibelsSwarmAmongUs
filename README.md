# LibelsSwarmAmongUs
Repository to perform analysis on pamphlets

A multi-source research pipeline for historical texts.

## Initial sources

- Dutch Pamphlets Online (DUPO)
  - Existing PDFs organized in year folders
  - Progressive OCR
  - DUPO identifier
  - Knuttel catalogue number extracted from the first page

- Text Creation Partnership (TCP)
  - Raw English text
  - Supplied bibliographic metadata
  - No OCR required

## Shared processing

Both sources use the same:

- relevance assessment
- fixed categories
- short topic descriptions
- summaries for relevant documents
- SQLite database
- full-text search and review interface

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

histtext doctor
pytest
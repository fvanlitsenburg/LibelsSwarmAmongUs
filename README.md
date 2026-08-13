# LibelsSwarmAmongUs

A research pipeline for processing and analysing historical pamphlets from two online collections: the Knuttel catalogue or 'Dutch Pamphlets Online' from Briel ('DUPO'), and the TCP Early English Texts.

The pipeline is set up to:
* transcribe the text of a pamphlet (relevant for DUPO only, as TCP is already transcribed)
* assign the text to a relevant topic, or declare it irrelevant, with instructions in `relevance_criteria.txt`. While these are set up for my personal research topic (pamphlet wars between the Dutch Republic and England), editing these could make the pipeline relevant for other topics too
* provide a summary of each text

## Sources

### Dutch Pamphlets Online (DUPO)
https://www.kb.nl/bron/dutch-pamphlets-online

Currently implemented.

Because of terms and conditions, the analysis is performed on selected pamphlets downloaded to a personal computer. Their path must be specified in env at `LSAU_DUPO_ROOT`. The path must be subdivided into a folder per year.

Folders are transcribed and then given a relevance assessment and summary.

### Text Creation Partnership (TCP)

Planned / partially scaffolded.

- Raw English text
- Supplied bibliographic metadata
- No OCR required
- Will feed into the same relevance and analysis pipeline as DUPO

## Processing pipeline

For DUPO documents:

1. Discover and register PDFs.
2. Inspect PDF page counts and embedded text.
3. OCR an initial batch of pages.
4. Assess relevance progressively.
5. Stop processing documents judged irrelevant.
6. Complete OCR for relevant documents.
7. Run a final full-text assessment.
8. Store category, topic, relevance explanation, and summary.

OCR text is stored separately from model-generated analysis so that the same
transcription can be reused without paying for OCR again.

## Database

The project uses PostgreSQL.

The database stores:

- documents
- DUPO metadata
- TCP metadata
- page-level OCR/text units
- progressive relevance assessments
- final document analyses

Database schema changes are managed with Alembic.

## Development setup

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

### 3. Start PostgreSQL

Start PostgresqL

```bash
docker compose up -d database
```

Confirm it's running and check logs:

```bash
docker compose ps
docker compose logs -f database
```

#### 4. Apply database migrations

```bash
alembic upgrade head
```

#### 5. Run tests

```bash
ruff check .
pytest
```

## Processing DUPO documents
### Progressive relevance

Process one relevance batch for eligible documents:

```bash
python scripts/run_dupo_batch.py relevance --limit 10
```

Preview the documents first without making API calls:

```bash
python scripts/run_dupo_batch.py relevance --limit 10 --dry-run
```

### Complete OCR

Complete OCR for documents already judged relevant:

```bash
python scripts/run_dupo_batch.py ocr --limit 10
```

### Final full-text assessment

Run final analysis for completely OCR'd documents:

```bash
python scripts/run_dupo_batch.py final --limit 10
```

### Full pipeline

Process documents through progressive relevance, OCR where required, and final assessment:

```bash
python scripts/run_dupo_batch.py pipeline --limit 10
```

Use a dry run before large batches:

```bash
python scripts/run_dupo_batch.py pipeline --limit 10 --dry-run
```

Processing is sequential and resumable. OCR pages are committed individually,
so an interrupted run can continue without repeating already stored OCR.

## Review interface

Start the local Streamlit interface with:

```bash
streamlit run ui/app.py --server.address 127.0.0.1
```

Then open:

http://localhost:8501

The interface provides:

* document filtering and ordering
* browsing previously processed documents
* complete OCR transcription
* relevance-assessment history
* summaries and document classifications
* controls for launching processing batches

The interface is currently intended for local use only.
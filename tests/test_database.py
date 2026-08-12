"""Tests for the shared PostgreSQL data model."""

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from historical_text_pipeline.db.models import (
    Document,
    DupoMetadata,
    TcpMetadata,
)
from historical_text_pipeline.domain import Source


def test_database_schema_is_created(db_engine: Engine) -> None:
    table_names = set(inspect(db_engine).get_table_names())

    assert table_names == {
        "documents",
        "document_analyses",
        "document_text_units",
        "dupo_metadata",
        "relevance_assessments",
        "tcp_metadata",
    }


def test_dupo_identifiers_are_separate(db_engine: Engine) -> None:
    document = Document(
        source=Source.DUPO,
        source_record_id="KB0_KB01436",
        source_path="/collection/1651/document.pdf",
        source_filename="document.pdf",
        language="nl",
        year=1651,
        title="Example Dutch pamphlet",
        dupo=DupoMetadata(
            dupo_id="KB0_KB01436",
            knuttel_number="1436",
        ),
    )

    with Session(db_engine) as session:
        session.add(document)
        session.commit()

    with Session(db_engine) as session:
        saved = session.scalar(
            select(Document).where(
                Document.source_record_id == "KB0_KB01436"
            )
        )

        assert saved is not None
        assert saved.dupo is not None
        assert saved.dupo.dupo_id == "KB0_KB01436"
        assert saved.dupo.knuttel_number == "1436"


def test_tcp_metadata_is_stored(db_engine: Engine) -> None:
    document = Document(
        source=Source.TCP,
        source_record_id="A12345",
        language="en",
        year=1650,
        source_date="1650?",
        title="Example English text",
        author="Example Author",
        tcp=TcpMetadata(
            tcp_id="A12345",
            eebo_id="99876543",
            vid="123456",
            stc="STC 1234",
            source_status="Free",
            source_terms=["trade", "religion"],
            source_pages="32",
        ),
    )

    with Session(db_engine) as session:
        session.add(document)
        session.commit()

    with Session(db_engine) as session:
        saved = session.scalar(
            select(Document).where(
                Document.source_record_id == "A12345"
            )
        )

        assert saved is not None
        assert saved.tcp is not None
        assert saved.tcp.tcp_id == "A12345"
        assert saved.tcp.source_terms == ["trade", "religion"]
        assert saved.dupo is None
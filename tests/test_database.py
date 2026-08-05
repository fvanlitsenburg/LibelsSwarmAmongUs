from pathlib import Path

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from historical_text_pipeline.db.base import Base
from historical_text_pipeline.db.models import (
    Document,
    DupoMetadata,
    TcpMetadata,
)
from historical_text_pipeline.domain import Source


def make_test_engine(database_path: Path):
    return create_engine(f"sqlite:///{database_path}")


def test_database_schema_is_created(tmp_path: Path) -> None:
    engine = make_test_engine(tmp_path / "schema.sqlite3")

    Base.metadata.create_all(engine)

    table_names = set(inspect(engine).get_table_names())

    assert table_names == {
        "documents",
        "document_text_units",
        "dupo_metadata",
        "relevance_assessments",
        "tcp_metadata",
    }


def test_dupo_identifiers_remain_separate(tmp_path: Path) -> None:
    engine = make_test_engine(tmp_path / "dupo.sqlite3")
    Base.metadata.create_all(engine)

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

    with Session(engine) as session:
        session.add(document)
        session.commit()

    with Session(engine) as session:
        saved = session.scalar(
            select(Document).where(
                Document.source_record_id == "KB0_KB01436"
            )
        )

        assert saved is not None
        assert saved.dupo is not None
        assert saved.dupo.dupo_id == "KB0_KB01436"
        assert saved.dupo.knuttel_number == "1436"


def test_tcp_metadata_is_stored(tmp_path: Path) -> None:
    engine = make_test_engine(tmp_path / "tcp.sqlite3")
    Base.metadata.create_all(engine)

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

    with Session(engine) as session:
        session.add(document)
        session.commit()

    with Session(engine) as session:
        saved = session.scalar(
            select(Document).where(Document.source_record_id == "A12345")
        )

        assert saved is not None
        assert saved.tcp is not None
        assert saved.tcp.tcp_id == "A12345"
        assert saved.tcp.source_terms == ["trade", "religion"]
        assert saved.dupo is None
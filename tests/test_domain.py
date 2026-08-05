from historical_text_pipeline.domain import (
    ClassificationStatus,
    DocumentMetadata,
    RelevanceStatus,
    Source,
)


def test_dupo_identifiers_are_stored_separately() -> None:
    document = DocumentMetadata(
        source=Source.DUPO,
        source_record_id="KB0_KB01436",
        dupo_id="KB0_KB01436",
        knuttel_number="1436",
        year=1651,
        title="Example Dutch pamphlet",
    )

    assert document.dupo_id == "KB0_KB01436"
    assert document.knuttel_number == "1436"
    assert document.source_record_id == "KB0_KB01436"
    assert document.relevance_status == RelevanceStatus.NOT_ASSESSED
    assert (
        document.classification_status
        == ClassificationStatus.NOT_CLASSIFIED
    )


def test_tcp_metadata_can_be_normalized() -> None:
    document = DocumentMetadata(
        source=Source.TCP,
        source_record_id="A12345",
        tcp_id="A12345",
        eebo_id="99876543",
        vid="123456",
        stc="STC 1234",
        author="Example Author",
        source_date="1650?",
        year=1650,
        title="Example English text",
        source_terms=["trade", "religion"],
        source_pages="32",
    )

    assert document.source == Source.TCP
    assert document.tcp_id == "A12345"
    assert document.source_terms == ["trade", "religion"]
    assert document.dupo_id is None
    assert document.knuttel_number is None
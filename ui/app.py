"""Local interface for browsing and processing historical documents."""

import os
import signal
import subprocess
import sys
import time
from html import escape
from pathlib import Path

import streamlit as st
from sqlalchemy import and_, select
from sqlalchemy.orm import aliased, selectinload

from historical_text_pipeline.db.models import (
    Document,
    DocumentAnalysis,
    DocumentProviderState,
    DocumentTextUnit,
    RelevanceAssessment,
)
from historical_text_pipeline.domain import (
    AnalysisProvider,
    ClassificationStatus,
    RelevanceStatus,
    Source,
)

from historical_text_pipeline.batch import (
    DupoBatchStage,
    get_dupo_batch_document_ids,
)
from historical_text_pipeline.db.session import (
    get_session_factory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = REPOSITORY_ROOT / ".runs"
BATCH_SCRIPT = (
    REPOSITORY_ROOT
    / "scripts"
    / "run_dupo_batch.py"
)

PAGE_SIZE = 50


def apply_layout_styles() -> None:
    """Apply small layout adjustments to the local interface."""

    st.markdown(
        """
        <style>
        .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-top: 3rem;
            padding-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def display_value(value: object) -> str:
    """Return a readable value for strings and enum members."""

    if value is None:
        return ""

    enum_value = getattr(value, "value", None)

    if isinstance(enum_value, str):
        return enum_value

    return str(value)


ORDER_LABELS = {
    "id": "Document ID",
    "year": "Year",
    "title": "Title",
    "source": "Source",
    "openai_relevance": "OpenAI relevance",
    "anthropic_relevance": "Anthropic relevance",
    "classification": "Text status",
    "processing": "Processing status",
}


def get_processing_status_options() -> list[str]:
    """Return processing statuses currently present in the database."""

    session_factory = get_session_factory()

    with session_factory() as session:
        statuses = list(
            session.scalars(
                select(Document.processing_status)
                .where(
                    Document.processing_status.is_not(None),
                    Document.processing_status != "",
                )
                .distinct()
                .order_by(Document.processing_status)
            )
        )

    return statuses


def search_documents(
    *,
    source: str,
    openai_relevance_status: str,
    anthropic_relevance_status: str,
    classification_status: str,
    processing_status: str,
    year_from: int | None,
    year_to: int | None,
    order_by: str,
    order_direction: str,
    page_number: int,
) -> tuple[list[Document], bool]:
    """Search, filter, and order documents."""

    session_factory = get_session_factory()

    openai_state = aliased(
        DocumentProviderState,
        name="openai_state",
    )

    anthropic_state = aliased(
        DocumentProviderState,
        name="anthropic_state",
    )

    statement = (
        select(Document)
        .outerjoin(
            openai_state,
            and_(
                openai_state.document_id == Document.id,
                openai_state.provider
                == AnalysisProvider.OPENAI.value,
            ),
        )
        .outerjoin(
            anthropic_state,
            and_(
                anthropic_state.document_id == Document.id,
                anthropic_state.provider
                == AnalysisProvider.ANTHROPIC.value,
            ),
        )
        .options(
            selectinload(Document.dupo)
        )
    )

    conditions = []

    if openai_relevance_status == "not_assessed":
        conditions.append(
            openai_state.id.is_(None)
        )
    elif openai_relevance_status != "any":
        conditions.append(
            openai_state.relevance_status
            == openai_relevance_status
        )

    if anthropic_relevance_status == "not_assessed":
        conditions.append(
            anthropic_state.id.is_(None)
        )
    elif anthropic_relevance_status != "any":
        conditions.append(
            anthropic_state.relevance_status
            == anthropic_relevance_status
        )

    if source != "any":
        conditions.append(
            Document.source == Source(source)
        )

    if classification_status != "any":
        conditions.append(
            Document.classification_status
            == ClassificationStatus(
                classification_status
            )
        )

    if processing_status != "any":
        conditions.append(
            Document.processing_status
            == processing_status
        )

    if year_from is not None:
        conditions.append(
            Document.year >= year_from
        )

    if year_to is not None:
        conditions.append(
            Document.year <= year_to
        )

    if conditions:
        statement = statement.where(*conditions)

    order_columns = {
        "id": Document.id,
        "year": Document.year,
        "title": Document.title,
        "source": Document.source,
        "openai_relevance": (
            openai_state.relevance_status
        ),
        "anthropic_relevance": (
            anthropic_state.relevance_status
        ),
        "classification": (
            Document.classification_status
        ),
        "processing": Document.processing_status,
    }
    order_columns = {
        "id": Document.id,
        "year": Document.year,
        "title": Document.title,
        "source": Document.source,
        "relevance": Document.relevance_status,
        "classification": Document.classification_status,
        "processing": Document.processing_status,
        "category": Document.primary_category,
        "topic": Document.topic,
    }

    order_column = order_columns[order_by]

    if order_direction == "ascending":
        main_order = order_column.asc().nulls_last()
        tie_breaker = Document.id.asc()
    else:
        main_order = order_column.desc().nulls_last()
        tie_breaker = Document.id.desc()

    statement = (
        statement
        .order_by(
            main_order,
            tie_breaker,
        )
        .offset(page_number * PAGE_SIZE)
        .limit(PAGE_SIZE + 1)
    )

    with session_factory() as session:
        documents = list(
            session.scalars(statement)
        )

    has_next_page = len(documents) > PAGE_SIZE

    return documents[:PAGE_SIZE], has_next_page


def document_table_rows(
    documents: list[Document],
) -> list[dict[str, object]]:
    """Prepare searchable document rows."""

    if not documents:
        return []

    session_factory = get_session_factory()

    document_ids = [
        document.id
        for document in documents
    ]

    with session_factory() as session:
        states = list(
            session.scalars(
                select(DocumentProviderState).where(
                    DocumentProviderState.document_id.in_(
                        document_ids
                    )
                )
            )
        )

    states_by_document = {
        (
            state.document_id,
            state.provider,
        ): state
        for state in states
    }

    rows: list[dict[str, object]] = []

    for document in documents:
        openai_state = states_by_document.get(
            (
                document.id,
                AnalysisProvider.OPENAI.value,
            )
        )

        anthropic_state = states_by_document.get(
            (
                document.id,
                AnalysisProvider.ANTHROPIC.value,
            )
        )

        rows.append(
            {
                "ID": document.id,
                "Year": document.year,
                "Source": display_value(
                    document.source
                ),
                "Title": (
                    document.title
                    or document.source_filename
                    or ""
                ),
                "OpenAI": (
                    display_value(
                        openai_state.relevance_status
                    )
                    if openai_state is not None
                    else "not assessed"
                ),
                "Anthropic": (
                    display_value(
                        anthropic_state.relevance_status
                    )
                    if anthropic_state is not None
                    else "not assessed"
                ),
                "Text": display_value(
                    document.classification_status
                ),
                "Processing": (
                    document.processing_status or ""
                ),
                "Pages": (
                    f"{document.units_processed or 0}/"
                    f"{document.total_units or '?'}"
                ),
            }
        )

    return rows

def load_document_bundle(
    document_id: int,
) -> tuple[
    Document | None,
    dict[str, DocumentProviderState],
    dict[str, DocumentAnalysis],
    list[RelevanceAssessment],
    list[DocumentTextUnit],
]:
    """Load one document and both providers' results."""

    session_factory = get_session_factory()

    with session_factory() as session:
        document = session.scalar(
            select(Document)
            .options(
                selectinload(Document.dupo)
            )
            .where(
                Document.id == document_id
            )
        )

        if document is None:
            return None, {}, {}, [], []

        states = list(
            session.scalars(
                select(DocumentProviderState).where(
                    DocumentProviderState.document_id
                    == document_id
                )
            )
        )

        provider_states = {
            state.provider: state
            for state in states
        }

        stored_analyses = list(
            session.scalars(
                select(DocumentAnalysis)
                .where(
                    DocumentAnalysis.document_id
                    == document_id
                )
                .order_by(
                    DocumentAnalysis.created_at.desc(),
                    DocumentAnalysis.id.desc(),
                )
            )
        )

        # DocumentAnalysis is append-only. Pick the newest
        # analysis for each provider.
        analyses: dict[str, DocumentAnalysis] = {}

        for analysis in stored_analyses:
            if analysis.provider not in analyses:
                analyses[analysis.provider] = analysis

        assessments = list(
            session.scalars(
                select(RelevanceAssessment)
                .where(
                    RelevanceAssessment.document_id
                    == document_id
                )
                .order_by(
                    RelevanceAssessment.provider,
                    RelevanceAssessment.sequence_number.desc(),
                )
            )
        )

        text_units = list(
            session.scalars(
                select(DocumentTextUnit)
                .where(
                    DocumentTextUnit.document_id
                    == document_id,
                    DocumentTextUnit.unit_type == "page",
                )
                .order_by(
                    DocumentTextUnit.unit_number
                )
            )
        )

    return (
        document,
        provider_states,
        analyses,
        assessments,
        text_units,
    )

def build_full_transcription(
    text_units: list[DocumentTextUnit],
) -> str:
    """Join all OCR pages with visible page indicators."""

    sections: list[str] = []

    for text_unit in text_units:
        page_text = text_unit.text.strip()

        if not page_text:
            page_text = "[No OCR text stored for this page.]"

        sections.append(
            "\n".join(
                [
                    "=" * 72,
                    f"PDF PAGE {text_unit.unit_number}",
                    "=" * 72,
                    "",
                    page_text,
                ]
            )
        )

    return "\n\n".join(sections)

def show_provider_analysis(
            *,
            label: str,
            state: DocumentProviderState | None,
            analysis: DocumentAnalysis | None,
        ) -> None:
            """Display one provider's current and final result."""

            st.markdown(f"#### {label}")

            relevance = (
                state.relevance_status
                if state is not None
                else "not assessed"
            )

            score = (
                analysis.relevance_score
                if analysis is not None
                else state.relevance_score
                if state is not None
                else None
            )

            confidence = (
                analysis.confidence
                if analysis is not None
                else state.confidence
                if state is not None
                else None
            )

            category = (
                analysis.primary_category
                if analysis is not None
                else state.primary_category
                if state is not None
                else None
            )

            topic = (
                analysis.topic
                if analysis is not None
                else state.topic
                if state is not None
                else None
            )

            explanation = (
                analysis.relevance_explanation
                if analysis is not None
                else state.relevance_reason
                if state is not None
                else None
            )

            st.write(
                f"**Relevance:** "
                f"{display_value(relevance)}"
            )
            st.write(
                "**Relevance score:** "
                + (
                    f"{score:.2f}"
                    if score is not None
                    else "—"
                )
            )
            st.write(
                "**Confidence:** "
                + (
                    f"{confidence:.2f}"
                    if confidence is not None
                    else "—"
                )
            )
            st.write(
                f"**Category:** {category or '—'}"
            )
            st.write(
                f"**Topic:** {topic or '—'}"
            )

            st.markdown("**Summary**")

            st.write(
                analysis.summary
                if (
                    analysis is not None
                    and analysis.summary
                )
                else "No final summary has been stored."
            )

            st.markdown("**Relevance explanation**")

            st.write(
                explanation
                or "No relevance explanation has been stored."
            )


def show_document_details(
    document_id: int,
) -> None:
    """Display one selected document in the right-hand frame."""

    (
        document,
        provider_states,
        analyses,
        assessments,
        text_units,
    ) = load_document_bundle(document_id)
    
    openai_state = provider_states.get(
    AnalysisProvider.OPENAI.value
    )
    anthropic_state = provider_states.get(
        AnalysisProvider.ANTHROPIC.value
    )

    openai_analysis = analyses.get(
        AnalysisProvider.OPENAI.value
    )
    anthropic_analysis = analyses.get(
        AnalysisProvider.ANTHROPIC.value
    )

    if document is None:
        st.error(
            f"Document {document_id} no longer exists."
        )
        return

    title = (
        document.title
        or document.source_filename
        or f"Document {document.id}"
    )

    st.subheader(title)

    fact_columns = st.columns(4)

    facts = [
        ("ID", document.id),
        ("Year", document.year or "Unknown"),
        (
            "Text",
            display_value(
                document.classification_status
            ),
        ),
        (
            "Pages",
            (
                f"{document.units_processed or 0}/"
                f"{document.total_units or '?'}"
            ),
        ),
    ]

    for column, (label, value) in zip(
        fact_columns,
        facts,
        strict=True,
    ):
        column.markdown(
            f"**{label}:** {value}"
        )

    overview_tab, text_tab, assessment_tab = st.tabs(
        [
            "Overview",
            "Complete transcription",
            "Assessment history",
        ]
    )

    with overview_tab:
        st.markdown("#### Catalogue")

        catalogue_columns = st.columns(2)

        with catalogue_columns[0]:
            st.write(
                f"**Source:** "
                f"{display_value(document.source)}"
            )
            st.write(
                f"**Filename:** "
                f"{document.source_filename or '—'}"
            )
            st.write(
                f"**Author:** "
                f"{document.author or '—'}"
            )

        with catalogue_columns[1]:
            st.write(
                f"**Processing status:** "
                f"{document.processing_status or '—'}"
            )

            if document.dupo is not None:
                st.write(
                    f"**DUPO ID:** "
                    f"{document.dupo.dupo_id or '—'}"
                )
                st.write(
                    f"**Knuttel number:** "
                    f"{document.dupo.knuttel_number or '—'}"
                )

        st.divider()

        provider_columns = st.columns(2)

        with provider_columns[0]:
            show_provider_analysis(
                label="OpenAI",
                state=openai_state,
                analysis=openai_analysis,
            )

        with provider_columns[1]:
            show_provider_analysis(
                label="Anthropic",
                state=anthropic_state,
                analysis=anthropic_analysis,
            )
            

        if document.dupo is not None:
            st.write(
                f"**DUPO ID:** "
                f"{document.dupo.dupo_id or '—'}"
            )
            st.write(
                f"**Knuttel number:** "
                f"{document.dupo.knuttel_number or '—'}"
            )

        st.markdown("#### Summary")
        st.write(
            document.summary
            or "No final summary has been stored."
        )

        st.markdown("#### Relevance explanation")
        st.write(
            document.relevance_reason
            or "No relevance explanation has been stored."
        )

        if document.error_message:
            st.error(document.error_message)

    with text_tab:
        if not text_units:
            st.info("No OCR text has been stored.")
        else:
            full_text = build_full_transcription(
                text_units
            )

            total_characters = sum(
                len(text_unit.text or "")
                for text_unit in text_units
            )

            st.caption(
                f"{len(text_units)} stored pages · "
                f"{total_characters:,} OCR characters"
            )

            with st.container(
                height=750,
                border=True,
            ):
                st.markdown(
                    (
                        "<div style='"
                        "white-space: pre-wrap;"
                        "overflow-wrap: anywhere;"
                        "font-family: ui-monospace, "
                        "SFMono-Regular, Menlo, Monaco, "
                        "Consolas, Liberation Mono, monospace;"
                        "font-size: 0.9rem;"
                        "line-height: 1.5;"
                        "'>"
                        f"{escape(full_text)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    with assessment_tab:
        if not assessments:
            st.info(
                "No relevance assessments have been stored."
            )
        else:
            assessment_rows = [
            {
                "Provider": (
                    assessment.provider.title()
                ),
                "Assessment": (
                    assessment.sequence_number
                ),
                "Pages": assessment.units_processed,
                "Decision": display_value(
                    assessment.decision
                ),
                "Score": (
                    assessment.relevance_score
                ),
                "Confidence": (
                    assessment.confidence
                ),
                "Category": (
                    assessment.primary_category or ""
                ),
                "Topic": assessment.topic or "",
                "Reason": assessment.reason or "",
            }
            for assessment in assessments
        ]

            st.dataframe(
                assessment_rows,
                width="stretch",
                hide_index=True,
                height=430,
            )

def show_document_navigation(
    *,
    selected_document_id: int,
    document_ids: list[int],
    has_next_page: bool,
) -> None:
    """Move between adjacent search results."""

    current_index = document_ids.index(
        selected_document_id
    )

    can_go_previous = (
        current_index > 0
        or st.session_state.document_page > 0
    )

    can_go_next = (
        current_index < len(document_ids) - 1
        or has_next_page
    )

    previous_column, position_column, next_column = (
        st.columns([1, 2, 1])
    )

    previous_clicked = previous_column.button(
        "← Previous",
        disabled=not can_go_previous,
        width="stretch",
    )

    position_column.markdown(
        (
            "<div style='text-align:center; padding-top:0.45rem'>"
            f"Result {current_index + 1} "
            f"of {len(document_ids)} on this page"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    next_clicked = next_column.button(
        "Next →",
        disabled=not can_go_next,
        width="stretch",
    )

    if previous_clicked:
        if current_index > 0:
            st.session_state.selected_document_id = (
                document_ids[current_index - 1]
            )
        else:
            st.session_state.document_page -= 1
            st.session_state.pending_document_position = (
                "last"
            )

        st.rerun()

    if next_clicked:
        if current_index < len(document_ids) - 1:
            st.session_state.selected_document_id = (
                document_ids[current_index + 1]
            )
        else:
            st.session_state.document_page += 1
            st.session_state.pending_document_position = (
                "first"
            )

        st.rerun()

def show_document_browser() -> None:
    """Render search on the left and the selected document on the right."""

    if "document_page" not in st.session_state:
        st.session_state.document_page = 0

    if "document_table_version" not in st.session_state:
        st.session_state.document_table_version = 0

    left_column, right_column = st.columns(
        [1, 1],
        gap="large",
    )

    processing_status_options = (
        get_processing_status_options()
    )

    with left_column, st.container(border=True):
        st.subheader("Search documents")

        with st.form("document_filters"):
            first_filter_row = st.columns(4)

            source = first_filter_row[0].selectbox(
                "Source",
                options=[
                    "any",
                    *[
                        item.value
                        for item in Source
                    ],
                ],
                format_func=lambda value: (
                    "Any"
                    if value == "any"
                    else value.upper()
                ),
            )

            relevance_options = [
                "any",
                "not_assessed",
                RelevanceStatus.RELEVANT.value,
                RelevanceStatus.UNCERTAIN.value,
                RelevanceStatus.IRRELEVANT.value,
            ]

            openai_relevance_status = (
                first_filter_row[1].selectbox(
                    "OpenAI relevance",
                    options=relevance_options,
                    format_func=lambda value: (
                        "Any"
                        if value == "any"
                        else value.replace("_", " ").title()
                    ),
                )
            )

            anthropic_relevance_status = (
                first_filter_row[2].selectbox(
                    "Anthropic relevance",
                    options=relevance_options,
                    format_func=lambda value: (
                        "Any"
                        if value == "any"
                        else value.replace("_", " ").title()
                    ),
                )
            )

            classification_status = (
                first_filter_row[3].selectbox(
                    "Text status",
                    options=[
                        "any",
                        *[
                            status.value
                            for status in ClassificationStatus
                        ],
                    ],
                    format_func=lambda value: (
                        "Any"
                        if value == "any"
                        else value.replace("_", " ").title()
                    ),
                )
            )

            second_filter_row = st.columns(4)

            processing_status = second_filter_row[0].selectbox(
                "Processing",
                options=[
                    "any",
                    *processing_status_options,
                ],
                format_func=lambda value: (
                    "Any"
                    if value == "any"
                    else value.replace("_", " ").title()
                ),
            )

            year_from_value = int(
                second_filter_row[1].number_input(
                    "Year from",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Use 0 for no lower limit.",
                )
            )

            year_to_value = int(
                second_filter_row[2].number_input(
                    "Year to",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Use 0 for no upper limit.",
                )
            )

            ordering_options = [
                f"{field}:ascending"
                for field in ORDER_LABELS
            ] + [
                f"{field}:descending"
                for field in ORDER_LABELS
            ]

            ordering = second_filter_row[3].selectbox(
                "Ordering",
                options=ordering_options,
                index=ordering_options.index(
                    "id:descending"
                ),
                format_func=lambda value: (
                    f"{ORDER_LABELS[value.split(':', 1)[0]]} — "
                    f"{value.split(':', 1)[1].title()}"
                ),
            )

            submitted = st.form_submit_button(
                "Apply filters",
                width="stretch",
            )
            
        order_by, order_direction = ordering.split(
            ":",
            maxsplit=1,
        )

        if submitted:
            st.session_state.document_page = 0
            st.session_state.selected_document_id = None
            st.session_state.document_table_version += 1

        year_from = (
            year_from_value
            if year_from_value > 0
            else None
        )

        year_to = (
            year_to_value
            if year_to_value > 0
            else None
        )

        documents, has_next_page = search_documents(
            source=source,
            openai_relevance_status=(
                openai_relevance_status
            ),
            anthropic_relevance_status=(
                anthropic_relevance_status
            ),
            classification_status=(
                classification_status
            ),
            processing_status=processing_status,
            year_from=year_from,
            year_to=year_to,
            order_by=order_by,
            order_direction=order_direction,
            page_number=(
                st.session_state.document_page
            ),
        )

        if not documents:
            st.info(
                "No documents match these filters."
            )
            return

        document_ids = [
            document.id
            for document in documents
        ]

        pending_position = st.session_state.get(
            "pending_document_position"
        )

        if pending_position == "first":
            st.session_state.selected_document_id = (
                document_ids[0]
            )

        elif pending_position == "last":
            st.session_state.selected_document_id = (
                document_ids[-1]
            )

        if pending_position is not None:
            del st.session_state[
                "pending_document_position"
            ]

        selected_document_id = (
            st.session_state.get(
                "selected_document_id"
            )
        )

        if selected_document_id not in document_ids:
            selected_document_id = document_ids[0]
            st.session_state.selected_document_id = (
                selected_document_id
            )

        st.caption(
            f"Results page "
            f"{st.session_state.document_page + 1}"
        )

        rows = document_table_rows(documents)

        table_event = st.dataframe(
            rows,
            key=(
                "document_table_"
                f"{st.session_state.document_table_version}_"
                f"{st.session_state.document_page}"
            ),
            width="stretch",
            height=535,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_order=[
                "ID",
                "Year",
                "Title",
                "OpenAI",
                "Anthropic",
                "Text",
            ],
        )

        if table_event.selection.rows:
            selected_index = (
                table_event.selection.rows[0]
            )

            selected_document_id = (
                rows[selected_index]["ID"]
            )

            st.session_state.selected_document_id = (
                selected_document_id
            )

        previous_page_column, next_page_column = (
            st.columns(2)
        )

        if previous_page_column.button(
            "Previous results",
            disabled=(
                st.session_state.document_page == 0
            ),
            width="stretch",
        ):
            st.session_state.document_page -= 1
            st.session_state.pending_document_position = (
                "first"
            )
            st.rerun()

        if next_page_column.button(
            "Next results",
            disabled=not has_next_page,
            width="stretch",
        ):
            st.session_state.document_page += 1
            st.session_state.pending_document_position = (
                "first"
            )
            st.rerun()

    with right_column, st.container(border=True):
        show_document_navigation(
            selected_document_id=(
                selected_document_id
            ),
            document_ids=document_ids,
            has_next_page=has_next_page,
        )

        st.divider()
        
        show_document_details(
            selected_document_id
        )

def eligible_document_preview(
    stage: DupoBatchStage,
    limit: int,
    start_id: int | None,
) -> list[dict[str, object]]:
    """Return a preview of documents eligible for a run."""

    session_factory = get_session_factory()

    with session_factory() as session:
        document_ids = get_dupo_batch_document_ids(
            session,
            stage=stage,
            limit=limit,
            start_id=start_id,
        )

        if not document_ids:
            return []

        documents = list(
            session.scalars(
                select(Document)
                .where(Document.id.in_(document_ids))
            )
        )

    by_id = {
        document.id: document
        for document in documents
    }

    return [
        {
            "ID": document_id,
            "Year": by_id[document_id].year,
            "Title": (
                by_id[document_id].title
                or by_id[document_id].source_filename
                or ""
            ),
            "Relevance": display_value(
                by_id[document_id].relevance_status
            ),
            "Text": display_value(
                by_id[document_id].classification_status
            ),
            "Processing": (
                by_id[document_id].processing_status or ""
            ),
        }
        for document_id in document_ids
    ]


def active_batch_process() -> subprocess.Popen | None:
    """Return the current running batch process."""

    process = st.session_state.get(
        "batch_process"
    )

    if process is None:
        return None

    if process.poll() is not None:
        return None

    return process


def start_batch_process(
    *,
    stage: DupoBatchStage,
    limit: int,
    start_id: int | None,
) -> None:
    """Launch the existing batch runner in the background."""

    RUN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    log_path = (
        RUN_DIRECTORY
        / f"{timestamp}-{stage.value}.log"
    )

    command = [
        sys.executable,
        "-u",
        str(BATCH_SCRIPT),
        stage.value,
        "--limit",
        str(limit),
    ]

    if start_id is not None:
        command.extend(
            [
                "--start-id",
                str(start_id),
            ]
        )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )

    st.session_state.batch_process = process
    st.session_state.batch_log_path = str(
        log_path
    )


def show_run_log(log_path: Path) -> None:
    """Display the tail of a batch log."""

    if not log_path.exists():
        st.info("The selected log does not exist.")
        return

    contents = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    st.code(
        contents[-40_000:],
        language="text",
    )


def show_batch_runs() -> None:
    """Render batch preview and execution controls."""

    st.header("Runs")

    st.write(
        "`relevance` processes unresolved documents; "
        "`ocr` completes relevant documents only; "
        "`final` assesses complete OCR without a final summary."
    )

    with st.form("batch_configuration"):
        stage = st.selectbox(
            "Stage",
            options=list(DupoBatchStage),
            format_func=lambda value: {
                DupoBatchStage.RELEVANCE: (
                    "Relevance — unresolved documents"
                ),
                DupoBatchStage.OCR: (
                    "OCR — relevant documents only"
                ),
                DupoBatchStage.FINAL: (
                    "Final assessment — complete OCR"
                ),
                DupoBatchStage.PIPELINE: (
                "Full pipeline — relevance → OCR → final"
                ),
            }[value],
        )

        limit = int(
            st.number_input(
                "Maximum documents",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
            )
        )

        start_id_value = int(
            st.number_input(
                "Starting document ID",
                min_value=0,
                value=0,
                step=1,
                help="Use 0 to start from the first eligible ID.",
            )
        )

        preview_submitted = (
            st.form_submit_button(
                "Preview run"
            )
        )

    if preview_submitted:
        st.session_state.batch_configuration = {
            "stage": stage,
            "limit": limit,
            "start_id": (
                start_id_value
                if start_id_value > 0
                else None
            ),
        }

    configuration = st.session_state.get(
        "batch_configuration"
    )

    process = st.session_state.get(
        "batch_process"
    )

    if process is not None:
        return_code = process.poll()

        if return_code is None:
            st.warning(
                f"Batch process {process.pid} is running."
            )

            stop_column, refresh_column = st.columns(
                [1, 4]
            )

            if stop_column.button(
                "Stop run",
                type="secondary",
            ):
                try:
                    os.killpg(
                        process.pid,
                        signal.SIGINT,
                    )
                except ProcessLookupError:
                    pass

                st.rerun()

            refresh_column.button("Refresh status")

        elif return_code == 0:
            st.success("The most recent batch completed.")

        else:
            st.error(
                "The most recent batch ended with "
                f"exit code {return_code}."
            )

    if configuration is not None:
        preview_rows = eligible_document_preview(
            stage=configuration["stage"],
            limit=configuration["limit"],
            start_id=configuration["start_id"],
        )

        st.subheader("Eligible documents")

        if preview_rows:
            st.dataframe(
                preview_rows,
                width="stretch",
                hide_index=True,
            )

            if st.button(
                "Start batch",
                type="primary",
                disabled=(
                    active_batch_process()
                    is not None
                ),
            ):
                start_batch_process(
                    stage=configuration["stage"],
                    limit=configuration["limit"],
                    start_id=configuration["start_id"],
                )

                st.rerun()

        else:
            st.info(
                "No documents are currently eligible "
                "for this stage."
            )

    st.subheader("Run logs")

    RUN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs = sorted(
        RUN_DIRECTORY.glob("*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not logs:
        st.info("No batch logs have been created.")
        return

    current_log = st.session_state.get(
        "batch_log_path"
    )

    default_index = 0

    if current_log:
        current_path = Path(current_log)

        if current_path in logs:
            default_index = logs.index(
                current_path
            )

    selected_log = st.selectbox(
        "Log",
        options=logs,
        index=default_index,
        format_func=lambda path: path.name,
    )

    show_run_log(selected_log)


def main() -> None:
    """Render the local application."""

    st.set_page_config(
        page_title="Libels Swarm Among Us",
        page_icon="📜",
        layout="wide",
    )

    apply_layout_styles()

    st.sidebar.title("Libels Swarm Among Us")

    workspace = st.sidebar.radio(
        "Workspace",
        options=[
            "documents",
            "runs",
        ],
        format_func=lambda value: {
            "documents": "Documents",
            "runs": "Processing runs",
        }[value],
    )

    st.sidebar.divider()

    process = st.session_state.get(
        "batch_process"
    )

    if (
        process is not None
        and process.poll() is None
    ):
        st.sidebar.warning(
            f"Batch process {process.pid} is running."
        )

    if workspace == "documents":
        show_document_browser()
    else:
        show_batch_runs()

if __name__ == "__main__":
    main()

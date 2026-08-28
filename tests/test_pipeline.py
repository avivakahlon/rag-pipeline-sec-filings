"""
Tests for the RAG pipeline.
Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


# --- ingest.py ---

def test_chunking_produces_output():
    from ingest import chunk_documents
    from langchain_core.documents import Document

    docs = [Document(page_content="word " * 200)]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_chunking_respects_chunk_size():
    from ingest import chunk_documents
    from langchain_core.documents import Document
    docs = [Document(page_content="word " * 200)]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    # allow a little slack since the splitter breaks on word boundaries, not mid-word
    assert all(len(c.page_content) <= 120 for c in chunks)


def test_chunking_empty_document_list():
    from ingest import chunk_documents

    chunks = chunk_documents([])
    assert chunks == []


# --- retrieve.py: near-duplicate detection ---

def test_near_duplicate_detects_identical_text():
    from retrieve import _is_near_duplicate

    seen = ["Mastercard is a technology company in the global payments industry."]
    duplicate = "Mastercard is a technology company in the global payments industry."
    assert _is_near_duplicate(duplicate, seen) is True


def test_near_duplicate_allows_distinct_text():
    from retrieve import _is_near_duplicate

    seen = ["Mastercard is a technology company in the global payments industry."]
    distinct = "JPMorgan faces significant cybersecurity risks from third-party vendors."
    assert _is_near_duplicate(distinct, seen) is False


def test_near_duplicate_empty_seen_list():
    from retrieve import _is_near_duplicate

    assert _is_near_duplicate("any text here", []) is False


def test_retrieve_rejects_empty_question():
    from retrieve import retrieve

    with pytest.raises(ValueError):
        retrieve("")

    with pytest.raises(ValueError):
        retrieve("   ")


def test_load_vector_store_missing_raises():
    from embed import load_vector_store

    with pytest.raises(FileNotFoundError):
        load_vector_store(persist_directory="data/does_not_exist")


# --- generate.py ---

def test_prompt_template_formats():
    from generate import PROMPT_TEMPLATE

    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE


def test_build_context_joins_documents():
    from generate import build_context
    from langchain_core.documents import Document

    docs = [Document(page_content="First chunk."), Document(page_content="Second chunk.")]
    context = build_context(docs)
    assert "First chunk." in context
    assert "Second chunk." in context
    assert context.index("First chunk.") < context.index("Second chunk.")


def test_build_context_empty_list():
    from generate import build_context

    assert build_context([]) == ""
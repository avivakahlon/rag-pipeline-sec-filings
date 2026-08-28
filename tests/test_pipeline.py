"""
Basic sanity tests for the RAG pipeline.
Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_chunking_produces_output():
    from ingest import chunk_documents
    from langchain.schema import Document

    docs = [Document(page_content="word " * 200)]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_prompt_template_formats():
    from generate import PROMPT_TEMPLATE

    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE

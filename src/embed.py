"""
Embed document chunks and populate the Chroma vector store.

Usage:
    python src/embed.py
"""
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # swap for OpenAI embeddings if preferred


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks, persist_directory: str = CHROMA_DIR):
    """Embed chunks and persist to a Chroma vector store."""
    if not chunks:
        raise ValueError(
            "No document chunks to embed. Check that data/raw/ contains source "
            "files and that ingest.py loaded them correctly."
        )
    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_DIR):
    """Load an existing Chroma vector store."""
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"No vector store found at '{persist_directory}'. "
            "Run `python src/embed.py` first to build it."
        )
    embeddings = get_embedding_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


if __name__ == "__main__":
    from ingest import load_documents, chunk_documents

    source_dir = Path("data/raw/")
    if not source_dir.exists() or not any(source_dir.iterdir()):
        print(f"Error: '{source_dir}' is missing or empty. Add source documents before running this script.")
        sys.exit(1)

    docs = load_documents(str(source_dir))
    if not docs:
        print(f"Error: no documents were loaded from '{source_dir}'. Check the file formats match what ingest.py expects.")
        sys.exit(1)

    chunks = chunk_documents(docs)

    try:
        build_vector_store(chunks)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Vector store built at {CHROMA_DIR} with {len(chunks)} chunks")
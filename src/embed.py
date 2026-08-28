"""
Embed document chunks and populate the Chroma vector store.

Usage:
    python src/embed.py
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
CHROMA_DIR = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # swap for OpenAI embeddings if preferred


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks, persist_directory: str = CHROMA_DIR):
    """Embed chunks and persist to a Chroma vector store."""
    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_DIR):
    """Load an existing Chroma vector store."""
    embeddings = get_embedding_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


if __name__ == "__main__":
    from ingest import load_documents, chunk_documents

    docs = load_documents("data/raw/")
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print(f"Vector store built at {CHROMA_DIR} with {len(chunks)} chunks")

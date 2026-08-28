"""
Document ingestion: load raw documents and split into chunks.

Usage:
    python src/ingest.py --source data/raw/
"""
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(source_dir: str):
    """Load all documents from source_dir. Extend loader_cls for other file types."""
    loader = DirectoryLoader(source_dir, loader_cls=TextLoader)
    return loader.load()


def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to raw documents")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    docs = load_documents(args.source)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

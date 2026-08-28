"""
Retrieval logic: given a query, fetch the most relevant chunks.

Usage:
    python src/retrieve.py --question "your question"
"""
import argparse

from embed import load_vector_store


def retrieve(question: str, k: int = 4, persist_directory: str = "data/chroma_db"):
    """Return the top-k most relevant chunks for a question."""
    vector_store = load_vector_store(persist_directory)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    results = retrieve(args.question, args.k)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])

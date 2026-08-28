"""
Retrieval logic: given a query, fetch the most relevant, de-duplicated chunks.

Filings from the same company across different fiscal years often share large
blocks of boilerplate language (business overview, standard regulatory
disclosures). Without filtering, the retriever can return several near-
identical chunks for a single query, wasting slots that could hold more
diverse, relevant context. This module over-fetches candidates and filters
out near-duplicates before returning the final top-k.

Usage:
    python src/retrieve.py --question "your question"
"""
import argparse
import sys
from difflib import SequenceMatcher

from embed import load_vector_store

DUPLICATE_SIMILARITY_THRESHOLD = 0.9  # chunks above this similarity are treated as duplicates


def _is_near_duplicate(text: str, seen_texts: list, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    """Check whether text is a near-duplicate of anything already in seen_texts."""
    for seen in seen_texts:
        ratio = SequenceMatcher(None, text, seen).ratio()
        if ratio >= threshold:
            return True
    return False


def retrieve(question: str, k: int = 4, persist_directory: str = "data/chroma_db", overfetch_multiplier: int = 3):
    """
    Return the top-k most relevant, de-duplicated chunks for a question.

    Fetches a wider candidate pool (k * overfetch_multiplier) from the vector
    store, then filters out near-duplicate chunks before returning the first
    k unique results. This keeps result diversity high even when the source
    documents contain repeated boilerplate language.

    Raises:
        FileNotFoundError: if the vector store hasn't been built yet (propagated from embed.py).
        ValueError: if the question is empty.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = load_vector_store(persist_directory)
    candidate_k = k * overfetch_multiplier
    retriever = vector_store.as_retriever(search_kwargs={"k": candidate_k})
    candidates = retriever.invoke(question)

    unique_results = []
    seen_texts = []
    for doc in candidates:
        if not _is_near_duplicate(doc.page_content, seen_texts):
            unique_results.append(doc)
            seen_texts.append(doc.page_content)
        if len(unique_results) >= k:
            break

    return unique_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    try:
        results = retrieve(args.question, args.k)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not results:
        print("No relevant results found for that question. Try rephrasing it.")
        sys.exit(0)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
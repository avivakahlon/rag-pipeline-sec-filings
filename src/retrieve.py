"""
Retrieval logic: given a query, fetch the most relevant, de-duplicated chunks.

Filings from the same company across different fiscal years often share large
blocks of boilerplate language (business overview, standard regulatory
disclosures). Without filtering, the retriever can return several near-
identical chunks for a single query, wasting slots that could hold more
diverse, relevant context. This module over-fetches candidates and filters
out near-duplicates before returning the final top-k.

Before searching, the question is also expanded into filing-style language
via a quick LLM call. Plain-English questions like "What business segments
does Mastercard operate in?" don't always share vocabulary with how a 10-K
actually phrases that section (e.g. "segment reporting", "reportable
segments"). Expanding the query closes that vocabulary gap before the
embedding search runs.

Usage:
    python src/retrieve.py --question "your question"
"""
import argparse
import sys
from difflib import SequenceMatcher

from dotenv import load_dotenv
from openai import OpenAI

from embed import load_vector_store

load_dotenv()
client = OpenAI()

DUPLICATE_SIMILARITY_THRESHOLD = 0.9  # chunks above this similarity are treated as duplicates


def expand_query(question: str) -> str:
    """Rewrite the question into keyword-dense, filing-style search language.

    Falls back to the original question if the expansion call fails for any
    reason (network issue, API error), so retrieval never breaks because of
    this optional step.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You expand user questions into short, keyword-dense search "
                        "queries for semantic search over SEC 10-K filings. Include "
                        "synonyms and the exact terminology filings typically use "
                        "(e.g. 'segment reporting', 'reportable segments', 'operating "
                        "segments'). Return ONLY the expanded query text, nothing else, "
                        "under 30 words."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else question
    except Exception:
        return question


def _is_near_duplicate(text: str, seen_texts: list, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> bool:
    """Check whether text is a near-duplicate of anything already in seen_texts."""
    for seen in seen_texts:
        ratio = SequenceMatcher(None, text, seen).ratio()
        if ratio >= threshold:
            return True
    return False


def retrieve(
    question: str,
    k: int = 4,
    persist_directory: str = "data/chroma_db",
    overfetch_multiplier: int = 3,
    use_query_expansion: bool = True,
):
    """
    Return the top-k most relevant, de-duplicated chunks for a question.

    Fetches a wider candidate pool (k * overfetch_multiplier) from the vector
    store, then filters out near-duplicate chunks before returning the first
    k unique results. This keeps result diversity high even when the source
    documents contain repeated boilerplate language.

    If use_query_expansion is True (default), the question is first rewritten
    into filing-style search language before the embedding search runs.

    Raises:
        FileNotFoundError: if the vector store hasn't been built yet (propagated from embed.py).
        ValueError: if the question is empty.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    search_query = expand_query(question) if use_query_expansion else question

    vector_store = load_vector_store(persist_directory)
    candidate_k = k * overfetch_multiplier
    retriever = vector_store.as_retriever(search_kwargs={"k": candidate_k})
    candidates = retriever.invoke(search_query)

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
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    args = parser.parse_args()

    if not args.no_expand:
        print(f"Expanded query: {expand_query(args.question)}\n")

    try:
        results = retrieve(args.question, args.k, use_query_expansion=not args.no_expand)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not results:
        print("No relevant results found for that question. Try rephrasing it.")
        sys.exit(0)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])

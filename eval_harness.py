"""
eval_harness.py

Automated LLM-as-judge evaluation for a RAG pipeline.
Scores each answer for factual grounding against the retrieved chunks,
flags hallucinated or unsupported claims, and logs everything to CSV.

Assumes you already have a working `retrieve_and_answer(question)` function
in your existing pipeline (the one that does embedding search + GPT-4o-mini
generation). Import it below instead of the placeholder.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads OPENAI_API_KEY from your .env file

sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate import generate_answer
from retrieve import retrieve

client = OpenAI()  # reads OPENAI_API_KEY from your environment


def retrieve_and_answer(question: str) -> dict:
    """Uses your real pipeline: retrieve() gets the source chunks,
    generate_answer() produces the final answer text."""
    sources = retrieve(question)
    answer = generate_answer(question)
    chunks = [doc.page_content for doc in sources]
    return {"answer": answer, "retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# 2. Your test questions (the same 8 you used for the 78% chunk-relevance check)
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What are the main risk factors facing JPMorgan?",
    "What business segments does Mastercard operate in?",
    "What risks does Visa identify related to competition?",
    "How does JPMorgan describe its approach to cybersecurity risk?",
    "What does Mastercard say about regulatory risk?",
    "What is Visa's business model or how does it generate revenue?",
    "What operational risks does JPMorgan identify?",
    "What competitive pressures does Mastercard face?",
]


# ---------------------------------------------------------------------------
# 3. The judge: a second LLM call that grades the first one's output
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge for a RAG system.
You will be given a question, the text chunks that were retrieved to answer it,
and the answer that was generated.

Your job:
1. Check every factual claim in the answer against the retrieved chunks.
2. Score factual grounding from 1 to 5:
   5 = every claim is directly supported by the retrieved text
   3 = mostly supported, minor unsupported details
   1 = mostly unsupported or contradicts the retrieved text
3. List any specific claims in the answer that are NOT supported by the
   retrieved chunks (this includes claims that are technically true but
   not present in the given context, since that means the model likely
   used outside knowledge instead of the retrieval).

Respond ONLY in this JSON format, nothing else:
{
  "score": <int 1-5>,
  "unsupported_claims": ["claim 1", "claim 2"],
  "reasoning": "one sentence explanation"
}
"""

def judge_answer(question: str, retrieved_chunks: list[str], answer: str) -> dict:
    context = "\n\n---\n\n".join(retrieved_chunks)
    user_prompt = f"""QUESTION:
{question}

RETRIEVED CHUNKS:
{context}

GENERATED ANSWER:
{answer}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# 4. Run the full eval and log results
# ---------------------------------------------------------------------------
def run_eval(threshold: int = 3, output_path: str = "eval_results.csv"):
    rows = []
    flagged_count = 0

    for question in TEST_QUESTIONS:
        result = retrieve_and_answer(question)
        answer = result["answer"]
        chunks = result["retrieved_chunks"]

        judgment = judge_answer(question, chunks, answer)
        score = judgment["score"]
        unsupported = "; ".join(judgment.get("unsupported_claims", []))
        reasoning = judgment.get("reasoning", "")

        needs_review = score < threshold
        if needs_review:
            flagged_count += 1
            print(f"[NEEDS HUMAN REVIEW] score={score} | {question}")
            if unsupported:
                print(f"    Unsupported claims: {unsupported}")

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "num_chunks_retrieved": len(chunks),
            "retrieved_chunks_preview": " ||| ".join(c[:200] for c in chunks),
            "score": score,
            "unsupported_claims": unsupported,
            "reasoning": reasoning,
            "needs_review": needs_review,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    avg_score = sum(r["score"] for r in rows) / len(rows)
    print(f"\nDone. {len(rows)} questions evaluated, avg score {avg_score:.2f}/5, "
          f"{flagged_count} flagged for review. Results saved to {output_path}")


if __name__ == "__main__":
    run_eval()
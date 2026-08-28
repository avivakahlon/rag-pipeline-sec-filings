"""
End-to-end CLI: ask a question, get a grounded answer from the RAG pipeline.

Usage:
    python src/query.py --question "your question here"
"""
import argparse

from generate import generate_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    print(generate_answer(args.question))

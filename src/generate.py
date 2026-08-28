"""
Generation: combine retrieved context with an LLM call to answer the query.

Usage:
    python src/generate.py --question "your question"
"""
import argparse
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieve import retrieve

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(question: str, k: int = 4):
    """
    Retrieve relevant chunks, then generate a grounded answer.
    Plug in your LLM of choice here (OpenAI, Anthropic, local model via HuggingFace).
    """
    docs = retrieve(question, k)
    context = build_context(docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(formatted_prompt)
    return response.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    answer = generate_answer(args.question)
    print(answer)

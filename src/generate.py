"""
Generation: combine retrieved context with an LLM call to answer the query.

Usage:
    python src/generate.py --question "your question"
"""
import argparse
import sys

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

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
    Retrieve relevant chunks, then generate a grounded answer via gpt-4o-mini.

    Raises:
        FileNotFoundError / ValueError: propagated from retrieve() for a missing
            vector store or empty question.
        RuntimeError: for OpenAI API failures (auth, quota, network), with a
            clear explanation instead of a raw stack trace.
    """
    docs = retrieve(question, k)

    if not docs:
        return "No relevant context was found for that question. Try rephrasing it."

    context = build_context(docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    load_dotenv()

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(formatted_prompt)
        return response.content
    except ImportError:
        raise RuntimeError(
            "langchain-openai is not installed. Run `pip install -r requirements.txt`."
        )
    except Exception as e:
        error_str = str(e).lower()
        if "authentication" in error_str or "invalid_api_key" in error_str:
            raise RuntimeError(
                "OpenAI authentication failed. Check that OPENAI_API_KEY in your .env file "
                "is set to a real, valid key."
            )
        elif "insufficient_quota" in error_str or "rate limit" in error_str or "429" in error_str:
            raise RuntimeError(
                "OpenAI API quota/rate limit hit. Check your account's billing/usage at "
                "platform.openai.com, you may need to add a payment method or wait before retrying."
            )
        else:
            raise RuntimeError(f"OpenAI API call failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    try:
        answer = generate_answer(args.question)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(answer)

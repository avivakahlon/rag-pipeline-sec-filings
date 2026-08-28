"""
Streamlit front-end for the RAG pipeline.

Usage:
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate import generate_answer
from retrieve import retrieve

st.set_page_config(page_title="SEC Filing Q&A", page_icon="📊", layout="centered")

st.title("📊 SEC 10-K Filing Q&A")
st.caption(
    "A RAG pipeline over SEC 10-K filings from JPMorgan Chase, Visa, and Mastercard. "
    "Ask a question and get an answer grounded in the actual filing text, with sources shown below."
)

with st.sidebar:
    st.header("About this corpus")
    st.markdown(
        """
        **Companies:**
        - JPMorgan Chase (2025 10-K)
        - Visa (2024, 2025 10-Ks)
        - Mastercard (2024, 2025 10-Ks)

        **Stack:**
        - LangChain + Chroma
        - `all-MiniLM-L6-v2` embeddings (local)
        - `gpt-4o-mini` generation

        Data pulled from SEC EDGAR's public API.
        """
    )
    st.divider()
    st.caption("[View source on GitHub](https://github.com/avivakahlon/rag-pipeline-sec-filings)")

question = st.text_input(
    "Ask a question about these filings",
    placeholder="e.g. What are the main risk factors facing JPMorgan?",
)

col1, col2 = st.columns([1, 4])
with col1:
    submitted = st.button("Ask", type="primary")

if submitted and question:
    with st.spinner("Retrieving relevant filing sections and generating an answer..."):
        try:
            answer = generate_answer(question)
            sources = retrieve(question)
        except FileNotFoundError as e:
            st.error(f"Vector store not found. {e}")
            st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources")
        st.caption(f"{len(sources)} retrieved chunk(s), deduplicated")
        for i, doc in enumerate(sources):
            source_name = Path(doc.metadata.get("source", "unknown")).name
            with st.expander(f"Source {i+1}: {source_name}"):
                st.text(doc.page_content)

elif submitted and not question:
    st.warning("Enter a question first.")
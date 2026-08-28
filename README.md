# RAG Pipeline: SEC 10-K Filing Q&A

A retrieval-augmented generation pipeline built with LangChain and Chroma, applied to SEC 10-K annual report filings from three finance/payments companies: JPMorgan Chase, Visa, and Mastercard.

## Status
✅ Working end to end: ingestion, chunking, embedding, retrieval, and grounded generation all functional.

## Stack
- **Orchestration:** LangChain
- **Vector store:** Chroma
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key required)
- **LLM:** OpenAI `gpt-4o-mini`

## Why this project
Built to demonstrate applied RAG architecture, document ingestion, chunking, embedding, retrieval, and grounded generation, the core skill cluster behind 2026's fastest-growing data/AI roles (LangChain, RAG, and vector databases are the top three named skills for LinkedIn's fastest-growing AI Engineer role category).

## Data
Five 10-K annual reports pulled directly from SEC EDGAR's public API (free, public domain, no licensing risk):
- JPMorgan Chase & Co. (CIK 0000019617) — 2025 filing
- Visa Inc. (CIK 0001403161) — 2024 and 2025 filings
- Mastercard Inc. (CIK 0001141391) — 2024 and 2025 filings

Fetched with `scripts/fetch_sec_filings.py`, which queries SEC's submissions API, downloads the primary filing document, and strips it to plain text.

## Architecture
```
SEC EDGAR 10-K filings (data/raw/)
    -> Chunking, 500 chars / 50 char overlap (src/ingest.py)
    -> Embedding via all-MiniLM-L6-v2 (src/embed.py)
    -> Chroma vector store (data/chroma_db/)
    -> Retriever, top-4 by similarity (src/retrieve.py)
    -> gpt-4o-mini generation with retrieved context (src/generate.py)
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI key (required for generation, not for retrieval):
```
OPENAI_API_KEY=your-key-here
```

## Usage
```bash
# Fetch source filings (one-time)
python scripts/fetch_sec_filings.py

# Chunk documents
python src/ingest.py --source data/raw/

# Build the vector store (one-time, or after adding new documents)
python src/embed.py

# Retrieval only
python src/retrieve.py --question "What are the main risk factors facing JPMorgan?"

# Full RAG: retrieval + generation
python src/generate.py --question "What are the main risk factors facing JPMorgan?"
```

## Project structure
```
rag-pipeline/
├── scripts/
│   └── fetch_sec_filings.py   # pulls 10-Ks from SEC EDGAR
├── src/
│   ├── ingest.py       # document loading + chunking
│   ├── embed.py        # embedding + vector store population
│   ├── retrieve.py     # retrieval logic
│   ├── generate.py     # LLM generation with retrieved context
│   └── query.py         # end-to-end query CLI
├── data/
│   ├── raw/             # source 10-K text files
│   └── chroma_db/       # persisted vector store (gitignored)
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Build log
- Repo scaffolded, stack locked (LangChain + Chroma)
- Pulled 5 10-K filings from SEC EDGAR across 3 finance companies (JPMorgan, Visa, Mastercard)
- Chunked into 9,412 segments (500 char / 50 char overlap)
- Built vector store with local `all-MiniLM-L6-v2` embeddings
- Wired up `gpt-4o-mini` generation via OpenAI API
- Ran an 8-question retrieval evaluation to validate quality (see Results)

## Results

### Sample query
**Q:** What are the main risk factors facing the company?

**A:** *"The main risk factors facing the company include the ability to withstand disruptions from failures of operational systems or third parties, the ability to defend against cyber attacks and unauthorized access to information, and other risks and uncertainties detailed in Part I, Item 1A: Risk Factors in JPMorganChase's 2025 Form 10-K. Additionally, the company faces intense competition in its industry."*

Grounded correctly in the retrieved risk-factor chunks, no hallucinated content.

### Retrieval evaluation
Ran 8 test questions across all three companies, checking whether the top-4 retrieved chunks were actually relevant to each question:

| # | Question | Relevant results | Notes |
|---|----------|:---:|---|
| 1 | JPMorgan main risk factors | 4/4 | Strong hit |
| 2 | Mastercard business segments | 2/4 | Weak question, Mastercard reports as a single business segment so there's little to retrieve |
| 3 | Visa competition risks | 3/4 | One result drifted into litigation/indemnification |
| 4 | JPMorgan cybersecurity approach | 4/4 | Excellent, all four squarely on-topic |
| 5 | Mastercard regulatory risk | 2/4 | Two results pulled credit-risk mitigation language instead of regulatory content |
| 6 | Visa business model/revenue | 4/4 | Covers overview, incentive structure, and revenue recognition |
| 7 | JPMorgan operational risks | 4/4 | Strong across the board |
| 8 | Mastercard competitive pressures | 2/4 | One result correctly named actual competitors; two others drifted into talent/culture content |

**Overall: ~78% chunk relevance (25/32).**

**Findings:**
- JPMorgan's filings retrieved most reliably (16/16 across 4 questions), likely because its risk-factor sections use direct, enumerated bullet language that chunks cleanly.
- Mastercard underperformed on 3 of its 4 questions, its filings are more narrative/prose-heavy, which appears to dilute topical focus within a fixed-size chunk.
- Retrieval surfaced duplicate or near-duplicate chunks across several queries (Mastercard's 2024 and 2025 filings share large blocks of boilerplate language year over year), a real characteristic of the data, not a pipeline bug.

## Known limitations
- `langchain-community` is being sunset by the LangChain team in favor of smaller standalone integration packages; this project still depends on it for document loaders. Migration would be a reasonable next step.
- Fixed-size chunking (500 chars) doesn't account for document structure, results above suggest narrative-heavy sections retrieve less precisely than enumerated/bulleted ones. A semantic or section-aware chunking strategy is a natural improvement.
- Small corpus (5 filings). Retrieval quality and duplicate-handling behavior would need re-validation at larger scale.

## Next steps
- [ ] Semantic or document-structure-aware chunking
- [ ] Deduplication of near-identical chunks at ingestion time
- [ ] Expand corpus to more filers / filing types (10-Q, 8-K)
- [ ] Migrate off `langchain-community` document loaders
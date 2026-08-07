# TradeTell (IMC Prosperity): RAG Evaluation & Decoupling Architecture Roadmap

This document outlines the engineering plan for upgrading **TradeTell** (the AI assistant for IMC Prosperity) from an in-memory prototype to a decoupled, high-performance RAG application with continuous retrieval/generation evaluation.

---

## 1. Objectives

1. **Eliminate Cold-Start Overhead:** Replace in-memory Chroma DB rebuilding with disk-persisted vector store collections (`data/vectordb_persisted/`).
2. **Decouple Frontend & Backend:** Transition from a unified Streamlit application to a dedicated **FastAPI** REST/SSE backend and a **React/Vite** frontend.
3. **Automated RAG Evaluation:** Establish continuous quality benchmarks measuring Retrieval Context Precision, Answer Faithfulness, and Groundedness.

---

## 2. Phase 1: Persistent Vector Indexing

Currently, `src/rag/build_rag_system.py` processes raw Markdown files, Discord exports, and code samples into Chroma vector stores at runtime startup.

### Changes:
- Update `Chroma` initialization in `build_rag_system.py` to specify `persist_directory="data/vectordb_persisted"`.
- Implement a CLI indexing script (`python -m src.rag.index_corpus`) to pre-build vector collections offline.
- Modify runtime initialization to attempt loading existing disk collections before attempting re-indexing.

---

## 3. Phase 2: Decoupled FastAPI Backend

Separate UI state and streaming LLM inference from Python application code.

### Backend Architecture (FastAPI):
- **`/api/v1/health`**: Diagnostic check for vector database availability and Groq API key validity.
- **`/api/v1/chat`**: Standard JSON payload endpoint returning response, source document citations, and retrieval scores.
- **`/api/v1/chat/stream`**: Server-Sent Events (SSE) endpoint delivering real-time answer tokens to the frontend.
- **`/api/v1/trading/summarize`**: Async endpoint processing uploaded competition log CSVs/txt files.

---

## 4. Phase 3: Automated RAG Quality Evaluation Harness

Continuous quality gate running alongside pytest in CI (`pytest tests/test_rag_eval.py`).

### Evaluation Metrics:
1. **Context Recall & Precision:** Percentage of top-k retrieved chunks containing ground-truth answers for competition rules (e.g. position limits, trading round schedules).
2. **Answer Faithfulness:** LLM score assessing whether generated claims strictly derive from retrieved context without hallucination.
3. **Code Syntax Validity:** AST validation ensuring generated `Trader` classes compile without syntax errors.

---

## 5. Timeline & Milestones

- **Milestone 1:** Disk-persisted vector storage implementation & cold-start benchmark (<1s vs 15s+).
- **Milestone 2:** Pytest evaluation suite with reference 20-question ground truth battery.
- **Milestone 3:** FastAPI routes & SSE streaming implementation.

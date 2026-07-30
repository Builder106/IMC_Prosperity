# Contributing

Thanks for your interest in the IMC Prosperity Trading Assistant.

## Development setup

Python 3.9+ (the deploy runs on 3.14; 3.12 is a safe local choice).

```bash
git clone https://github.com/Builder106/IMC_Prosperity.git
cd IMC_Prosperity
uv sync --group dev
```

Create a `.env` with your Groq API key ([console.groq.com/keys](https://console.groq.com/keys)):

```
GROQ_API_KEY=your_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Tests

```bash
pytest
```

`pytest.ini` scopes collection to `tests/`, so the vendored `tools/` tree is
ignored. Add tests under `tests/` mirroring the module under test
(e.g. `tests/test_groq_llm.py`). Network calls — Groq and the embedding model —
must be mocked; the suite runs offline.

## Project guardrails

- **Pin dependencies only where a version is known to break.** `requirements.txt`
  pins the langchain 0.2.x stack, a recent OpenTelemetry, and `protobuf<7`;
  everything else is intentionally unpinned so the deploy resolver can pick
  wheels for its Python version. A full freeze broke the Streamlit Cloud install
  once — don't reintroduce blanket pins.
- **The vector store is built in-memory** (`Chroma.from_documents` with no
  `persist_directory`). It's rebuilt on every cold start under
  `@st.cache_resource`; don't reintroduce on-disk persistence — the SQLite path
  fails on Streamlit Cloud.
- **LLM calls go through `GroqRagChain`** (`src/rag/groq_llm.py`), behind the
  contract `invoke({"query": ...}) -> {"result", "source_documents"}`. Keep that
  contract so the backend stays swappable.
- **Secrets are never committed.** `GROQ_API_KEY` lives in `.env` locally or in
  Streamlit secrets on Cloud. The repo is public.

## Commit messages

Conventional-commit style: `type: summary` (`feat`, `fix`, `chore`, `data`,
`build`, `docs`). Imperative subject, ~72 chars max.

## Pull requests

1. Branch off `main`.
2. Keep `pytest` green and add tests for new behavior.
3. One logical change per PR; explain the *why*, not just the *what*.

## Out of scope

- **`tools/`** — a vendored third-party backtester clone. Gitignored; don't
  commit it or build features against it here.
- **`data/competition_rounds/`, `*.zip`** — local competition data dumps.
  Gitignored.
- **End-to-end / demo-video suite** — deliberately omitted. The app needs a Groq
  key and a runtime-built vector store, which makes headless E2E impractical;
  unit tests cover the logic instead.

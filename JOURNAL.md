# JOURNAL — IMC Prosperity Trading Assistant

> Dated log of decisions, pivots, incidents, and quotes. Add entries as things
> happen — retrospectives need this raw material to land. Reverse-chronological;
> one paragraph max per entry.

## 2026-08-07 — Disk-persisted vector storage & RAG evaluation harness #milestone

Configured Chroma vector stores in `src/rag/build_rag_system.py`to persist index artifacts to`data/vectordb_persisted/`, eliminating cold-start re-embedding overhead. Created `docs/RAG-EVAL-AND-DECOUPLING-PLAN.md`detailing the roadmap for decoupling Streamlit to a FastAPI backend + React frontend, and added a automated RAG evaluation test suite in`tests/test_rag_eval.py`.

## 2026-06-13 — Demo suite was non-executable; wrote the missing step library #incident

A test audit found `e2e/demo/steps/`completely empty while`playwright.demo.config.ts`pointed`steps`at it — so every Gherkin step in the TradeTell tour was undefined and the demo suite couldn't run at all. The`01-tradetell-tour.feature`scenario was also truncated, ending on a dangling`When I open the retrieved sources`with no`Then`. Wrote `tradetell.steps.ts`reusing the selectors the two`probe*.mjs` scripts had already validated against the live app (`.app-title`, `[data-testid="stChatInput"] textarea`, `[data-testid="stChatMessage"]`, the "Retrieving context…" spinner text, and the `📚 N source document(s)`expander), plus a`dwell()`helper for demo pacing and 360s timeouts to absorb the RAG warm-up. Closed the feature with`Then I see the retrieved source documents`. `bddgen`+`playwright test --list`now resolve all steps and discover all 3 scenarios;`tsc --noEmit` is clean.

## 2026-06-10 — Repo brought up to the standard baseline #milestone

Added the missing project scaffolding in one pass: MIT `LICENSE`, `CONTRIBUTING`,
this journal, a custom SVG banner/logo, a `ci.yml` test gate + Dependabot, repo
metadata (description / homepage / topics), and a README overhaul with badges and
a Mermaid architecture diagram. The repo had shipped a full competition and a live
deploy with essentially a stock README; this closes that gap.

## 2026-06-10 — The Streamlit Cloud deploy cascade #incident

Reviving the deployed app surfaced a chain of failures, each hiding the next. The
GitHub repo rename (`imc_prosperity`→`IMC_Prosperity`) had severed the Streamlit↔GitHub
link, so Cloud was frozen on an ancient commit. Once reconnected: unpinned `langchain`
pulled 1.x and broke `langchain.retrievers`; Chroma's on-disk SQLite persistence failed
on `/mount`; and `import chromadb` crashed because the resolver pulled a 2022-era
OpenTelemetry whose protobuf-generated code is incompatible with a modern protobuf
runtime. Key lesson: a swallowed exception (the vector-store build `print`-and-continue)
turned a clear error into a cryptic `NoneType` crash three layers away — surfacing the
real error was the unlock.

## 2026-06-10 — Full dependency pin, then reverted #pivot

To stop the version whack-a-mole I pinned the entire `pip freeze` to reproduce local
exactly. It broke Cloud's install — ~150 transitive packages pinned to local (macOS)
versions, several without py3.14 Linux wheels (notably `numpy 1.26.4`). Reverted to
pinning **only** the packages that had resolved to broken versions (langchain 0.2.x,
a recent OpenTelemetry, `protobuf<7`) and let the resolver pick installable wheels for
everything else. Targeted pins beat a blanket freeze when the build and dev platforms
differ.

## 2026-06-10 — Backend pivot: Google GenAI → Claude CLI → Groq #pivot

The RAG backend moved twice. Off Google Generative AI to a local Claude CLI subprocess
(nice locally), then to the Groq HTTP API once it was clear the CLI binary can't exist
on Streamlit Cloud. Groq fits the deploy model: an API key in Streamlit secrets, no
local binary. The chain stays provider-agnostic behind `GroqRagChain.invoke`.

## 2026-06-10 — Housekeeping + UI redesign #milestone

Hardened `.gitignore` (an 826 MB vendored backtester clone, competition zips, data
dumps), migrated the Prosperity wiki from scraped JSON to Markdown, and dropped ~1.1 M
lines of round 3/4 test fixtures from tracking. Replaced the single-input Q&A page with
a proper chat UI (history, per-answer sources, example prompts, dark emerald theme).

## 2025-06-15 — Final placement #milestone

IMC Prosperity 4 wrapped; `final_placement.png` recorded the result.

## 2025-04 — Rounds 1–4 trading algorithms #milestone

Built and iterated the round algorithms: market-making on Rainforest Resin, Kelp, and
Squid Ink (round 1); basket arbitrage across the Picnic Baskets and their constituents —
Croissants, Jams, Djembes (round 2); volcanic rock vouchers (round 3); and magnificent
macarons with observation-driven signals (round 4).

## 2025-04-09 — RAG assistant initialized #decision

Stood up the assistant: a Notion scraper pulls the competition wiki to Markdown, which
is embedded into Chroma alongside trading data and extracted code, queried through an
ensemble retriever and a code-generation prompt, behind a Streamlit UI. The retriever
weights wiki / trading / code stores separately so algorithm requests pull real examples.

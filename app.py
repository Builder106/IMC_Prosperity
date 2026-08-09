import os

# Chroma requires sqlite3 >= 3.35, but Streamlit Cloud ships an older system
# sqlite. Swap in pysqlite3 (a manylinux wheel) before anything imports chromadb.
# No-ops locally where pysqlite3 isn't installed (e.g. macOS).
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

from src.rag.build_rag_system import (
    create_combined_retriever,
    create_rag_chain,
    create_vector_stores,
    process_discord_data,
    process_notion_wiki_data,
    process_trading_data,
)

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
# Custom browser-tab favicon: the candlestick logo (assets/logo.png). Resolved
# relative to this file so it works regardless of the working directory the app
# is launched from (Streamlit Cloud runs from the repo root). Streamlit's
# page_icon expects a raster path/emoji — it doesn't render SVG favicons — so we
# point at the PNG, not logo.svg. Falls back to the chart emoji if the file is
# missing.
_FAVICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
st.set_page_config(
    page_title="IMC Prosperity Trading Assistant",
    page_icon=_FAVICON if os.path.exists(_FAVICON) else "📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      /* Tighten the top whitespace Streamlit leaves above the content. */
      .block-container { padding-top: 2.5rem; max-width: 960px; }

      /* Gradient wordmark for the header. */
      .app-title {
        font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #10B981 0%, #34D399 60%, #6EE7B7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
      }
      .app-subtitle { color: #8B98A5; font-size: 0.95rem; margin-bottom: 1.25rem; }

      /* Make the example-prompt buttons read as quiet suggestion chips. */
      section[data-testid="stSidebar"] .stButton > button {
        width: 100%; text-align: left; border: 1px solid #2A333D;
        background: #11181F; color: #C9D4DF; font-size: 0.85rem;
        padding: 0.5rem 0.75rem; white-space: normal; line-height: 1.3;
      }
      section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #10B981; color: #FFFFFF;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_PROMPTS = [
    "What products and position limits are introduced in Round 1?",
    "Write a market-making Trader class for RAINFOREST_RESIN.",
    "Explain a basket-arbitrage strategy for the PICNIC_BASKET products.",
    "How should I manage risk against position limits in my algorithm?",
]


# --------------------------------------------------------------------------- #
# RAG system (built once, cached for the session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading knowledge base and building the RAG system…")
def initialize_rag_system():
    notion_documents = process_notion_wiki_data()
    notion_documents.extend(process_discord_data())
    trading_documents = process_trading_data()
    notion_vectorstore, trading_vectorstore, code_vectorstore = create_vector_stores(
        notion_documents, trading_documents
    )
    retriever = create_combined_retriever(
        notion_vectorstore, trading_vectorstore, code_vectorstore
    )
    return create_rag_chain(retriever)


def render_sources(source_documents):
    """Render retrieved context inside a collapsed expander."""
    if not source_documents:
        return
    label = f"📚 {len(source_documents)} source document(s)"
    with st.expander(label):
        for i, doc in enumerate(source_documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            st.markdown(f"**{i}. `{os.path.basename(str(source))}`**")
            st.code(doc.page_content, language="text")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📈 Prosperity Assistant")
    st.caption("RAG-powered insights over IMC Prosperity wiki, Discord, and trading data.")

    with st.expander("About", expanded=False):
        st.markdown(
            "Ask questions about products, position limits, and strategies — or "
            "request a complete, executable `Trader` algorithm. Answers are grounded "
            "in the competition wiki, community Discord threads, and historical "
            "trading data via retrieval-augmented generation."
        )

    st.markdown("**Try asking**")
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=f"ex_{prompt}"):
            st.session_state.pending_prompt = prompt

    st.divider()
    if st.button("🗑️ Clear conversation", key="clear"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built for the IMC Prosperity trading competition.")


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
st.markdown('<div class="app-title">IMC Prosperity Trading Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask about products, position limits, and strategies — '
    "or request a ready-to-run trading algorithm.</div>",
    unsafe_allow_html=True,
)

# On Streamlit Cloud, secrets aren't automatically in os.environ; bridge the
# config keys across so the os.getenv-based model config picks them up. Locally
# this no-ops (no secrets file) and .env is used instead.
try:
    # HF_TOKEN is read by huggingface-hub when sentence-transformers pulls the
    # embedding model; bridging it authenticates the download (higher rate limit,
    # no "unauthenticated requests to the HF Hub" warning).
    for _key in ("GROQ_API_KEY", "LLM_MODEL", "LLM_TEMPERATURE", "GROQ_TIMEOUT_SECONDS", "EMBEDDING_MODEL", "HF_TOKEN"):
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
except ImportError:
    # st.secrets may not be available in all contexts
    pass

try:
    rag_chain = initialize_rag_system()
except Exception as exc:  # surface setup problems instead of a raw traceback
    st.error(
        "The assistant could not start. Check that GROQ_API_KEY is set (in "
        "Streamlit secrets or your .env file) and the knowledge base is available."
    )
    st.exception(exc)
    st.stop()
    raise  # Help Pyright understand execution stops here

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

# A clicked example prompt stands in for typed input on this rerun.
prompt = st.chat_input("Ask a question, or request a trading algorithm…")
pending = st.session_state.pop("pending_prompt", None)
if pending:
    prompt = pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer…"):
            try:
                result = rag_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = result.get("source_documents", [])
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Something went wrong while answering: `{exc}`"
                sources = []
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

import sys
import os
import streamlit as st

print(f"--- DEBUG: app.py execution started ---")
print(f"Current working directory (os.getcwd()): {os.getcwd()}")
print(f"Python sys.path:")
for p in sys.path:
    print(f"  - {p}")
print(f"--- END DEBUG ---")

from src.rag.build_rag_system import (
    process_notion_wiki_data,
    process_trading_data,
    create_vector_stores,
    create_combined_retriever,
    create_rag_chain
)

st.set_page_config(
    page_title="IMC Prosperity Trading Assistant",
    layout="wide"
)

@st.cache_resource
def initialize_rag_system():
    # Load data and create RAG system (only done once)
    notion_documents = process_notion_wiki_data()
    print(f"[DEBUG app.py] Number of Notion documents processed: {len(notion_documents)}")
    trading_documents = process_trading_data()
    print(f"[DEBUG app.py] Number of Trading documents processed: {len(trading_documents)}")
    notion_vectorstore, trading_vectorstore, code_vectorstore = create_vector_stores(
        notion_documents, trading_documents
    )
    retriever = create_combined_retriever(notion_vectorstore, trading_vectorstore, code_vectorstore)
    if retriever is None:
        print("[DEBUG app.py] Retriever is None before calling create_rag_chain.")
    else:
        print(f"[DEBUG app.py] Retriever type: {type(retriever)}")
    rag_chain = create_rag_chain(retriever)
    return rag_chain

# Initialize the system
rag_chain = initialize_rag_system()

# App UI
st.title("IMC Prosperity Trading Assistant")
st.markdown("""
Ask questions about IMC Prosperity trading data and get AI-powered insights.
""")

# User input
query = st.text_input("Enter your question:", key="query")

# Display results
if query:
    with st.spinner("Processing your question..."):
        result = rag_chain.invoke({"query": query})
        
    st.markdown("### Answer")
    st.markdown(result["result"])
    
    # Optional: Show sources/documents used
    with st.expander("View Source Documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}**")
            st.markdown(f"```\n{doc.page_content}\n```")
            st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
            st.divider()
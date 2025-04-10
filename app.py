import streamlit as st
from build_rag_system import (
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
    trading_documents = process_trading_data()
    notion_vectorstore, trading_vectorstore = create_vector_stores(
        notion_documents, trading_documents
    )
    retriever = create_combined_retriever(notion_vectorstore, trading_vectorstore)
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
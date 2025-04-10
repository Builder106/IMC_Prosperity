import os
import json
from pathlib import Path
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from process_trading_data import process_all_csv_files

# Load environment variables (for OpenAI API key)
load_dotenv()

# Directory settings
NOTION_WIKI_DIR = "prosperity_wiki"
TRADING_DATA_DIR = "round_1_island_data"
PROCESSED_TRADING_DATA_DIR = "processed_trading_data"
VECTOR_DB_DIR = "vectordb"

def process_notion_wiki_data(wiki_dir=NOTION_WIKI_DIR):
    """
    Process Notion Wiki JSON data
    
    Args:
        wiki_dir: Directory containing Notion Wiki data
        
    Returns:
        List of Document objects suitable for vector store
    """
    print(f"Processing Notion Wiki data from {wiki_dir}...")
    
    wiki_path = Path(wiki_dir)
    documents = []
    
    # Categories to process
    categories = ["about_prosperity", "e-learning_center", "rounds"]
    
    for category in categories:
        category_path = wiki_path / category
        
        if not category_path.exists():
            print(f"Category directory {category} not found")
            continue
            
        # Process each JSON file in the category directory
        for json_file in category_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract content from the JSON structure
                content = ""
                
                # Check if data is a list or dict and handle accordingly
                if isinstance(data, list):
                    # Handle list structure
                    for item in data:
                        if isinstance(item, dict):
                            if "title" in item:
                                content += f"# {item['title']}\n\n"
                            if "content" in item:
                                content += f"{item['content']}\n\n"
                            # Add other fields as needed
                else:
                    # Original code for dictionary structure
                    if "title" in data:
                        content += f"# {data['title']}\n\n"
                    
                    if "content_blocks" in data:
                        for block in data["content_blocks"]:
                            if "text" in block:
                                content += f"{block['text']}\n\n"
                            elif "code" in block:
                                content += f"```\n{block['code']}\n```\n\n"
                
                # Create Document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(json_file),
                        "category": category,
                        "type": "notion_wiki",
                        "title": data.get("title", "")
                    }
                )
                
                documents.append(doc)
                print(f"Processed {json_file.name}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    print(f"Processed {len(documents)} Notion Wiki documents")
    return documents

def process_trading_data():
    """
    Process trading data CSV files into documents
    
    Returns:
        List of Document objects from trading data
    """
    # First process CSVs to create JSON documents
    print(f"Processing trading data from {TRADING_DATA_DIR}...")
    json_documents = process_all_csv_files(TRADING_DATA_DIR, PROCESSED_TRADING_DATA_DIR)
    
    # Convert to LangChain Documents
    documents = []
    for json_doc in json_documents:
        doc = Document(
            page_content=json_doc["content"],
            metadata=json_doc["metadata"]
        )
        documents.append(doc)
    
    print(f"Processed {len(documents)} trading data documents")
    return documents

def create_vector_stores(notion_documents, trading_documents):
    """
    Create vector stores for both document types
    
    Args:
        notion_documents: List of Notion Wiki documents
        trading_documents: List of trading data documents
        
    Returns:
        Tuple of (notion_vectorstore, trading_vectorstore)
    """
    print("Creating vector stores...")
    
    # Initialize the embedding model with Google's generative embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )
    
    # Create directories for vector stores
    os.makedirs(f"{VECTOR_DB_DIR}/notion", exist_ok=True)
    os.makedirs(f"{VECTOR_DB_DIR}/trading", exist_ok=True)
    
    # Create text splitter for longer documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # Split notion documents if needed
    split_notion_docs = text_splitter.split_documents(notion_documents)
    
    # Create notion vector store
    notion_vectorstore = Chroma.from_documents(
        documents=split_notion_docs,
        embedding=embeddings,
        persist_directory=f"{VECTOR_DB_DIR}/notion"
    )
    
    # Create trading data vector store
    trading_vectorstore = Chroma.from_documents(
        documents=trading_documents,
        embedding=embeddings,
        persist_directory=f"{VECTOR_DB_DIR}/trading"
    )
    
    # Persist the vector stores
    notion_vectorstore.persist()
    trading_vectorstore.persist()
    
    print("Vector stores created and persisted")
    return notion_vectorstore, trading_vectorstore

def create_combined_retriever(notion_vectorstore, trading_vectorstore):
    """
    Create an ensemble retriever that combines both vector stores
    
    Args:
        notion_vectorstore: Vector store for Notion Wiki data
        trading_vectorstore: Vector store for trading data
        
    Returns:
        Ensemble retriever
    """
    print("Creating combined retriever...")
    
    notion_retriever = notion_vectorstore.as_retriever(search_kwargs={"k": 3})
    trading_retriever = trading_vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create an ensemble retriever with weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[notion_retriever, trading_retriever],
        weights=[0.7, 0.3]  # Adjust the weights as needed
    )
    
    return ensemble_retriever

def create_rag_chain(retriever):
    """
    Create a RAG chain with the retriever
    
    Args:
        retriever: Retriever to use in the chain
        
    Returns:
        RAG chain
    """
    print("Creating RAG chain...")
    
    # Create a prompt template
    rag_prompt_template = """
    You are a financial analysis assistant with expertise in trading data for IMC Prosperity.
    Use the following retrieved information to answer the user's question.
    If you can't answer based on the retrieved information, say so.

    Retrieved information:
    {context}

    User question: {question}

    Provide a detailed answer with any relevant trading insights:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=rag_prompt_template,
    )
    
    # Initialize Google Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.0,
        convert_system_message_to_human=True
    )
    
    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain

def main():
    """Main execution function"""
    # 1. Process notion wiki data
    notion_documents = process_notion_wiki_data()
    
    # 2. Process trading data
    trading_documents = process_trading_data()
    
    # 3. Create vector stores
    notion_vectorstore, trading_vectorstore = create_vector_stores(
        notion_documents, trading_documents
    )
    
    # 4. Create combined retriever
    retriever = create_combined_retriever(notion_vectorstore, trading_vectorstore)
    
    # 5. Create RAG chain
    rag_chain = create_rag_chain(retriever)
    
    # 6. Example query
    print("\nTesting the RAG system with an example query...")
    query = "What was the price trend for RAINFOREST_RESIN on day -1?"
    result = rag_chain.run(query)
    
    print("\nQuery:", query)
    print("\nResult:", result)
    
    print("\nRAG system is ready for use!")
    print("You can now query it with your own questions about IMC Prosperity trading data and Notion Wiki.")

if __name__ == "__main__":
    main()

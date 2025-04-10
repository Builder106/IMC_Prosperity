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
from process_trading_data import process_round_data, discover_rounds

# Load environment variables (for OpenAI API key)
load_dotenv()

# Directory settings
NOTION_WIKI_DIR = "prosperity_wiki"
TRADING_DATA_DIR = "trading_data"  # Updated to the main trading_data directory
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
                
                # Process the list structure with proper type handling
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Handle different content types
                            if "type" in item:
                                if item["type"].startswith("h"):
                                    # Handle headings (h1, h2, h3, etc.)
                                    heading_level = item["type"][1:]
                                    content += f"{'#' * int(heading_level)} {item['content']}\n\n"
                                elif item["type"] == "p":
                                    # Handle paragraphs
                                    content += f"{item['content']}\n\n"
                                elif item["type"] == "list" and "items" in item:
                                    # Handle lists with nested items
                                    content += process_list_items(item["items"], item.get("style", "bulleted"))
                            # Fallback for any other structure
                            elif "content" in item:
                                content += f"{item['content']}\n\n"
                else:
                    # Handle dictionary structure (if exists)
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
                        "title": extract_title(data)
                    }
                )
                
                documents.append(doc)
                print(f"Processed {json_file.name}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    print(f"Processed {len(documents)} Notion Wiki documents")
    return documents

def process_list_items(items, style="bulleted"):
    """Process nested list items and return formatted content"""
    result = ""
    for item in items:
        if isinstance(item, dict) and "content" in item:
            # Calculate indentation based on nesting level
            indent = "  " * (item.get("level", 0))
            # Add appropriate marker based on list style
            marker = "- " if style == "bulleted" else f"1. "
            result += f"{indent}{marker}{item['content']}\n"
    result += "\n"  # Add extra line after list
    return result

def extract_title(data):
    """Extract title from the data structure"""
    if isinstance(data, list):
        # Look for the first h1 element as title
        for item in data:
            if isinstance(item, dict) and item.get("type") == "h1" and "content" in item:
                return item["content"]
    elif isinstance(data, dict) and "title" in data:
        return data["title"]
    return ""

def process_trading_data():
    """
    Process trading data CSV files into documents
    
    Returns:
        List of Document objects from trading data
    """
    print(f"Processing trading data from {TRADING_DATA_DIR}...")
    
    all_documents = []
    
    # Discover available rounds
    available_rounds = discover_rounds(TRADING_DATA_DIR)
    print(f"Discovered rounds: {available_rounds}")
    
    # Process each available round
    for round_name in available_rounds:
        print(f"Processing {round_name}...")
        json_documents = process_round_data(round_name, TRADING_DATA_DIR)
        
        # Convert JSON documents to LangChain Document objects
        for json_doc in json_documents:
            doc = Document(
                page_content=json_doc["content"],
                metadata=json_doc["metadata"]
            )
            all_documents.append(doc)
    
    print(f"Processed a total of {len(all_documents)} trading data documents")
    return all_documents

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
    
    # Initialize vector stores as None
    notion_vectorstore = None
    trading_vectorstore = None
    
    # Only create notion vector store if there are documents
    if notion_documents:
        # Split notion documents if needed
        split_notion_docs = text_splitter.split_documents(notion_documents)
        
        if split_notion_docs:
            print(f"Creating notion vector store with {len(split_notion_docs)} documents")
            # Create notion vector store
            notion_vectorstore = Chroma.from_documents(
                documents=split_notion_docs,
                embedding=embeddings,
                persist_directory=f"{VECTOR_DB_DIR}/notion"
            )
    else:
        print("No notion documents to process")
    
    # Only create trading vector store if there are documents
    if trading_documents:
        print(f"Creating trading vector store with {len(trading_documents)} documents")
        # Create trading data vector store
        trading_vectorstore = Chroma.from_documents(
            documents=trading_documents,
            embedding=embeddings,
            persist_directory=f"{VECTOR_DB_DIR}/trading"
        )
    else:
        print("No trading documents to process")
    
    # Only persist vector stores if they exist
    if notion_vectorstore:
        notion_vectorstore.persist()
    if trading_vectorstore:
        trading_vectorstore.persist()
    
    print("Vector stores created and persisted")
    return notion_vectorstore, trading_vectorstore

def create_combined_retriever(notion_vectorstore, trading_vectorstore):
    print("Creating combined retriever...")
    
    # Handle case where one or both vector stores might be None
    retrievers = []
    weights = []
    
    if notion_vectorstore:
        notion_retriever = notion_vectorstore.as_retriever(search_kwargs={"k": 3})
        retrievers.append(notion_retriever)
        weights.append(0.7)
    
    if trading_vectorstore:
        trading_retriever = trading_vectorstore.as_retriever(search_kwargs={"k": 2})
        retrievers.append(trading_retriever)
        weights.append(0.3)
    
    # Normalize weights if we have at least one retriever
    if retrievers:
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Create an ensemble retriever with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )
        return ensemble_retriever
    else:
        print("Warning: No retrievers available. Cannot create ensemble retriever.")
        return None

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
        model="gemini-2.5-pro-exp-03-25",
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
    
    # 4. Create combined retriever if possible
    retriever = None
    if notion_vectorstore or trading_vectorstore:
        retriever = create_combined_retriever(notion_vectorstore, trading_vectorstore)
    
    # 5. Create RAG chain and start interactive query session if retriever exists
    if retriever:
        rag_chain = create_rag_chain(retriever)
        
        print("\nRAG system is ready for use!")
        print("You can now query it with your own questions about IMC Prosperity trading data and Notion Wiki.")
        print("Enter 'quit', 'exit', or 'q' to end the session.")
        
        # Interactive query loop
        while True:
            query = input("\nEnter your question: ")
            
            # Check for exit commands
            if query.lower() in ["quit", "exit", "q"]:
                print("Exiting RAG query session. Goodbye!")
                break
                
            if not query.strip():
                print("Please enter a valid question.")
                continue
                
            # Process the query
            try:
                print("\nProcessing your question...")
                result = rag_chain.run(query)
                
                print("\nAnswer:")
                print(result)
            except Exception as e:
                print(f"\nError processing your question: {e}")
                print("Please try again with a different question.")
    else:
        print("\nCould not create RAG system: No retrievers available")

if __name__ == "__main__":
    main()

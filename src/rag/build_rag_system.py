import os
import json
import re
from pathlib import Path
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from .process_raw_trading_data import process_round_data, discover_rounds

# Load environment variables
load_dotenv()

# Directory settings
# SCRIPT_DIR should be defined before it's used for other constants
SCRIPT_DIR = Path(__file__).resolve().parent  # Gets the directory of the current script (src/rag)
PROJECT_ROOT = SCRIPT_DIR.parent.parent    # Goes up two levels to the project root (imc_prosperity)

NOTION_WIKI_DIR = PROJECT_ROOT / "data" / "prosperity_wiki"
TRADING_DATA_DIR = PROJECT_ROOT / "data" / "trading_data"
VECTOR_DB_DIR = PROJECT_ROOT / "vectordb"

def process_notion_wiki_data(wiki_dir=NOTION_WIKI_DIR):
    """
    Process Notion Wiki JSON data
    
    Args:
        wiki_dir: Directory containing Notion Wiki data
        
    Returns:
        List of Document objects suitable for vector store
    """
    print(f"Processing Notion Wiki data from {wiki_dir}...")
    
    def load_code_file(file_path):
        """Load code from external file referenced in the JSON"""
        full_path = wiki_dir / file_path
        try:
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                print(f"Warning: Referenced code file not found: {full_path}")
                return None
        except Exception as e:
            print(f"Error loading code file {full_path}: {e}")
            return None
    
    def extract_code_languages(data):
        """Extract programming languages used in code blocks"""
        languages = set()
        
        # Handle list structure
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "code" and "language" in item:
                    languages.add(item["language"])
        
        # Handle dictionary structure
        elif isinstance(data, dict) and "content_blocks" in data:
            for block in data["content_blocks"]:
                if "code" in block and "language" in block:
                    languages.add(block["language"])
                    
        return list(languages) if languages else []
    
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
                                elif item["type"] == "code" and "file_path" in item:
                                    # Load external code file
                                    code_content = load_code_file(item["file_path"])
                                    if code_content:
                                        language = item.get("language", "")
                                        content += f"```{language}\n{code_content}\n```\n\n"
                                    else:
                                        # Fallback to preview if available
                                        preview = item.get("preview", "Code content unavailable")
                                        content += f"```\n{preview}\n```\n\n"
                                elif item["type"] == "code":
                                    # Handle inline code
                                    code = item.get("code", "")
                                    language = item.get("language", "")
                                    content += f"```{language}\n{code}\n```\n\n"
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
                                language = block.get("language", "")
                                content += f"```{language}\n{block['code']}\n```\n\n"
                            elif "file_path" in block:
                                # Load external code file
                                code_content = load_code_file(block["file_path"])
                                if code_content:
                                    language = block.get("language", "")
                                    content += f"```{language}\n{code_content}\n```\n\n"
                                else:
                                    # Fallback to preview if available
                                    preview = block.get("preview", "Code content unavailable")
                                    content += f"```\n{preview}\n```\n\n"
                
                # Create Document with enhanced metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(json_file),
                        "category": category,
                        "type": "notion_wiki",
                        "title": extract_title(data),
                        "contains_code": "```" in content,
                        "code_languages": extract_code_languages(data)
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

def extract_code_blocks(documents):
    """
    Extract code blocks from documents for specialized code search
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of Document objects containing only code blocks
    """
    code_docs = []
    code_pattern = r'```(?:(\w+))?\n(.*?)\n```'
    
    for doc in documents:
        # Use regex to find code blocks with optional language specification
        matches = re.findall(code_pattern, doc.page_content, re.DOTALL)
        
        if matches:
            for i, (language, code) in enumerate(matches):
                if code.strip():  # Only add non-empty code blocks
                    code_docs.append(Document(
                        page_content=code,
                        metadata={
                            **doc.metadata,
                            "content_type": "code_block",
                            "language": language if language else "unknown",
                            "block_index": i
                        }
                    ))
    
    return code_docs

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
        Tuple of (notion_vectorstore, trading_vectorstore, code_vectorstore)
    """
    print("Creating vector stores...")
    
    # Initialize the embedding model with Google's generative embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )
    
    # Create directories for vector stores
    os.makedirs(VECTOR_DB_DIR / "notion", exist_ok=True)
    os.makedirs(VECTOR_DB_DIR / "trading", exist_ok=True)
    os.makedirs(VECTOR_DB_DIR / "code", exist_ok=True)
    
    # Create text splitter optimized for code and general content
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",  # Default to Python handling
        chunk_size=1500,
        chunk_overlap=200,
    )
    
    # Initialize vector stores as None
    notion_vectorstore = None
    trading_vectorstore = None
    code_vectorstore = None
    
    # Helper function to ensure document is a proper Document object
    def ensure_document(item):
        if isinstance(item, Document):
            return item
        elif isinstance(item, tuple):
            # Handle tuple case - careful with extraction
            if len(item) >= 1:
                content = str(item[0]) if item[0] is not None else ""
                metadata = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
                return Document(page_content=content, metadata=metadata)
        elif hasattr(item, 'page_content'):
            # Handle case where it has page_content but isn't a Document instance
            metadata = item.metadata if hasattr(item, 'metadata') else {}
            return Document(page_content=item.page_content, metadata=metadata)
        else:
            # Last resort - convert to string
            return Document(page_content=str(item), metadata={})
    
    # Helper function to clean metadata to ensure it only contains simple types
    def clean_metadata(metadata):
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # Convert None to empty string
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                # Allow other primitive types as is
                cleaned[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                cleaned[key] = ", ".join(str(v) for v in value if v is not None)
            elif isinstance(value, dict):
                # Convert dictionaries to string representation
                cleaned[key] = str(value)
            else:
                # Convert any other type to string
                cleaned[key] = str(value)
        return cleaned
    
    # Only create notion vector store if there are documents
    if notion_documents:
        try:
            # Split notion documents if needed
            split_notion_docs = text_splitter.split_documents(notion_documents)
            
            if split_notion_docs:
                # Process all documents to ensure they're valid Document objects
                valid_notion_docs = []
                for item in split_notion_docs:
                    try:
                        doc = ensure_document(item)
                        valid_notion_docs.append(doc)
                    except Exception as e:
                        print(f"Error processing notion document: {e}")
                
                # Now clean and normalize the metadata
                filtered_notion_docs = []
                for doc in valid_notion_docs:
                    try:
                        # First create a document with cleaned metadata
                        cleaned_doc = Document(
                            page_content=doc.page_content,
                            metadata=clean_metadata(doc.metadata)
                        )
                        filtered_notion_docs.append(cleaned_doc)
                    except Exception as e:
                        print(f"Error cleaning notion document metadata: {e}")
                        # If cleaning fails, try filter_complex_metadata as fallback
                        try:
                            filtered_doc = filter_complex_metadata(doc)
                            filtered_notion_docs.append(filtered_doc)
                        except Exception as e2:
                            print(f"Error filtering notion document metadata: {e2}")
                            # Last resort: create a document with minimal metadata
                            minimal_metadata = {"source": doc.metadata.get("source", "unknown")} if hasattr(doc, "metadata") else {}
                            filtered_notion_docs.append(Document(page_content=doc.page_content, metadata=minimal_metadata))
                
                print(f"[DEBUG build_rag_system.py] Creating notion vector store with {len(filtered_notion_docs)} documents")
                # Create notion vector store
                notion_vectorstore = Chroma.from_documents(
                    documents=filtered_notion_docs,
                    embedding=embeddings,
                    persist_directory=str(VECTOR_DB_DIR / "notion")
                )
                
                # Extract code blocks for specialized code search
                code_blocks = extract_code_blocks(notion_documents)
                if code_blocks:
                    # Filter complex metadata structures from code blocks
                    filtered_code_blocks = []
                    for doc in code_blocks:
                        try:
                            filtered_code_blocks.append(filter_complex_metadata(ensure_document(doc)))
                        except Exception as e:
                            print(f"Error filtering code block metadata: {e}")
                    
                    print(f"[DEBUG build_rag_system.py] Creating specialized code vector store with {len(filtered_code_blocks)} code blocks")
                    code_vectorstore = Chroma.from_documents(
                        documents=filtered_code_blocks,
                        embedding=embeddings,
                        persist_directory=str(VECTOR_DB_DIR / "code")
                    )
        except Exception as e:
            print(f"Error processing notion documents: {e}")
    else:
        print("No notion documents to process")
    
    # Only create trading vector store if there are documents
    if trading_documents:
        try:
            # Split trading documents if needed
            split_trading_docs = text_splitter.split_documents(trading_documents)
            
            if split_trading_docs:
                # Process all documents to ensure they're valid Document objects
                valid_trading_docs = []
                for item in split_trading_docs:
                    try:
                        doc = ensure_document(item)
                        valid_trading_docs.append(doc)
                    except Exception as e:
                        print(f"Error processing trading document: {e}")
                
                # Now clean and normalize the metadata
                filtered_trading_docs = []
                for doc in valid_trading_docs:
                    try:
                        # First create a document with cleaned metadata
                        cleaned_doc = Document(
                            page_content=doc.page_content,
                            metadata=clean_metadata(doc.metadata)
                        )
                        filtered_trading_docs.append(cleaned_doc)
                    except Exception as e:
                        print(f"Error cleaning trading document metadata: {e}")
                        # If cleaning fails, try filter_complex_metadata as fallback
                        try:
                            filtered_doc = filter_complex_metadata(doc)
                            filtered_trading_docs.append(filtered_doc)
                        except Exception as e2:
                            print(f"Error filtering trading document metadata: {e2}")
                            # Last resort: create a document with minimal metadata
                            minimal_metadata = {"source": doc.metadata.get("source", "unknown")} if hasattr(doc, "metadata") else {}
                            filtered_trading_docs.append(Document(page_content=doc.page_content, metadata=minimal_metadata))
                
                print(f"[DEBUG build_rag_system.py] Creating trading vector store with {len(filtered_trading_docs)} documents")
                # Create trading data vector store
                trading_vectorstore = Chroma.from_documents(
                    documents=filtered_trading_docs,
                    embedding=embeddings,
                    persist_directory=str(VECTOR_DB_DIR / "trading")
                )
        except Exception as e:
            print(f"Error processing trading documents: {e}")
    else:
        print("No trading documents to process")
    
    # Documents are automatically persisted in Chroma 0.4.x+
    # No need to call persist() method anymore
    
    print("Vector stores created and persisted automatically")
    return notion_vectorstore, trading_vectorstore, code_vectorstore

def create_combined_retriever(notion_vectorstore, trading_vectorstore, code_vectorstore=None):
    print("[DEBUG build_rag_system.py] Creating combined retriever...")
    print(f"[DEBUG build_rag_system.py] Notion vectorstore is None: {notion_vectorstore is None}")
    print(f"[DEBUG build_rag_system.py] Trading vectorstore is None: {trading_vectorstore is None}")
    print(f"[DEBUG build_rag_system.py] Code vectorstore is None: {code_vectorstore is None}")
    
    # Handle case where one or both vector stores might be None
    retrievers = []
    weights = []
    
    if notion_vectorstore:
        notion_retriever = notion_vectorstore.as_retriever(search_kwargs={"k": 45})
        retrievers.append(notion_retriever)
        weights.append(0.2)
    
    if trading_vectorstore:
        trading_retriever = trading_vectorstore.as_retriever(search_kwargs={"k": 65})
        retrievers.append(trading_retriever)
        weights.append(0.3)
        
    if code_vectorstore:
        code_retriever = code_vectorstore.as_retriever(search_kwargs={"k": 50})
        retrievers.append(code_retriever)
        weights.append(0.5)
    
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
    
    # Create a prompt template with code generation instructions
    rag_prompt_template = """
    You are an expert algorithmic trading developer specializing in IMC Prosperity trading algorithms.
    Use the following retrieved information to generate a complete, executable trading algorithm.
    
    When asked to create a trading algorithm:
    1. Generate a complete `Trader` class with all necessary methods and proper implementation
    2. Include detailed docstrings and comments explaining the strategy
    3. Implement proper position management and risk controls based on position limits
    4. Format the output as a single Python file that can be executed directly
    5. Ensure the code follows best practices and handles edge cases
    
    If the retrieved information doesn't provide enough details for certain products, 
    use reasonable default strategies based on similar products and clearly mark these assumptions.
    
    Retrieved information:
    {context}
    
    User question: {question}
    
    Start by providing a brief overview of the strategy, then generate the complete algorithm as a Python file:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=rag_prompt_template,
    )
    
    # Initialize Google Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0.2  # Slightly increased temperature for more creative code generation
    )
    
    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return rag_chain

def main():
    """Main execution function"""
    # 1. Process notion wiki data
    notion_documents = process_notion_wiki_data()
    
    # 2. Process trading data
    trading_documents = process_trading_data()
    
    # 3. Create vector stores
    notion_vectorstore, trading_vectorstore, code_vectorstore = create_vector_stores(
        notion_documents, trading_documents
    )
    
    # 4. Create combined retriever if possible
    retriever = None
    if notion_vectorstore or trading_vectorstore or code_vectorstore:
        retriever = create_combined_retriever(notion_vectorstore, trading_vectorstore, code_vectorstore)
    
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
                # Using invoke() instead of run() to address deprecation warning
                response = rag_chain.invoke({"query": query})
                result = response.get("result", "No result found")
                
                print("\nAnswer:")
                print(result)
            except Exception as e:
                print(f"\nError processing your question: {e}")
                print("Please try again with a different question.")
    else:
        print("\nCould not create RAG system: No retrievers available")

if __name__ == "__main__":
    main()
"""
PDF Agent: A Chatbot for PDF Document Interaction

This program processes PDF documents, extracts metadata, 
and allows users to interact with the content through a chat interface. 
It utilizes OpenAI's language model to generate responses based on the 
conversation history and relevant document context. 
The workflow includes resetting a vector store for document 
embeddings, processing PDF files, and managing conversation memory.

Key Features:
- Load and process PDF documents.
- Extract metadata such as date and author.
- Maintain conversation history with memory management.
- Generate AI responses using document context and chat history.
"""

# Standard library imports
import os
import sys
import shutil
from typing import List

# Env imports and load env variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Langchain and Langtrace imports
from langtrace_python_sdk import langtrace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# 1. Configuration and Setup
def reset_vector_store(persist_directory):
    """Reset or initialize the vector store directory."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)  # Remove existing vector store directory
        #print(f"Vector store at '{persist_directory}' has been reset.")
    else:
        print(f"No existing vector store found at '{persist_directory}'.")

def extract_metadata(doc):
    """Extract metadata from document using LLM analysis."""
    system_prompt = """You are a metadata extraction assistant. Given a document, extract the following metadata:
    1. Date (if present)
    2. Author (if present)
    Return ONLY a JSON-like string in this exact format:
    {"date": "found_date or null", "author": "found_author or null"}"""
    
    # Prepare messages for the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract metadata from this text:\n\n{doc.page_content}")
    ]
    
    # Invoke the model to extract metadata
    response = model.invoke(messages)
    
    try:
        # Clean up the response to get just the JSON string
        import json
        json_str = response.content.strip()
        if json_str.startswith("```") and json_str.endswith("```"):
            json_str = json_str[3:-3].strip()  # Remove code block formatting
        metadata = json.loads(json_str)  # Parse the JSON string
        return metadata
    except Exception as e:
        print(f"Error parsing metadata: {e}")  # Handle JSON parsing errors
        return {"date": None, "author": None}

def view_chroma_database(vector_store):
    """Print the contents of the Chroma vector store."""
    count = vector_store._collection.count()  # Get the number of documents in the vector store
    print(f"Number of documents in the Chroma database: {count}")
    
    # Retrieve and print the documents using the get() method
    all_data = vector_store.get()  # Use the get() method to retrieve all documents
    for i in range(len(all_data["ids"])):
        doc_id = all_data["ids"][i]  # Get document ID
        content = all_data["documents"][i]  # Get document content
        print(f"\nDocument ID: {doc_id}, \nContent: {content}")

# 2. Document Processing
def process_documents(pdf_paths: List[str]):
    """Process PDF documents and create vector store."""
    # Load PDFs using the PyPDFLoader
    loaders = [PyPDFLoader(path) for path in pdf_paths]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())  # Load documents into the list

    # Determine chunk size and overlap based on document size
    if len(docs) < 5:  # Example condition for small documents
        chunk_size = 1000
        chunk_overlap = 50
    elif len(docs) < 20:  # Example condition for medium documents
        chunk_size = 500
        chunk_overlap = 100
    else:  # Large documents
        chunk_size = 200
        chunk_overlap = 50

    # print("# of docs: ", len(docs))
    # print("Chunk size: ", chunk_size)
    # print("Chunk overlap: ", chunk_overlap)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", " "]  # Define separators for splitting
    )
    splits = text_splitter.split_documents(docs)  # Split the loaded documents into chunks
    
    return docs, splits  # Return the original documents and their splits

# 3. Chat System Components
def create_augmented_response(state: MessagesState, vector_store):
    """
    Generate AI responses using a combination of conversation history and relevant document context.
    
    This function performs three main steps:
    1. Retrieves the latest user message from the conversation state
    2. Searches the vector store using Max Marginal Relevance (MMR) to find relevant and diverse document chunks.
    3. Combines document context, conversation history, and system prompt for the LLM
    
    Args:
        state (MessagesState): Current conversation state containing message history
        vector_store: Vector database containing embedded document chunks
    
    Returns:
        dict: Contains 'messages' key with the AI's response message
    
    Search Parameters:
        - Retrieves 3 most relevant chunks (k=3)
        - From initial pool of 6 chunks (fetch_k=6)
        - Using 70% relevance, 30% diversity weighting (lambda_mult=0.7)
    """
    # Get the latest question from the conversation state
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    
    # Retrieve relevant documents using Max Marginal Relevance
    results = vector_store.max_marginal_relevance_search(
        query=latest_msg,
        k=6,                # Number of documents to return
        fetch_k=10,        # Initial pool of documents to fetch 
        lambda_mult=0.7     # Diversity weight parameter
    )
    
    # Combine the content of the retrieved documents
    context = " ".join([doc.page_content for doc in results])
    
    # Create messages with system prompt, chat history, and context
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Use both the conversation history "
            "and the provided context to give accurate answers. "
            f"Context: {context}"
        ))
    ] + state["messages"]
    
    # Get response from the model
    response = model.invoke(messages)
    return {"messages": response}  # Return the AI's response

def setup_chat_workflow(vector_store):
    """Create and configure the chat workflow with memory management."""
    # Define the function that processes messages
    def process_messages(state: MessagesState):
        return create_augmented_response(state, vector_store)
    
    # Add node and edge to workflow
    workflow.add_node("chat", process_messages)
    workflow.add_edge(START, "chat")
    
    # Add memory management
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def chat_with_memory(question: str, app, thread_id: str = "default"):
    """Process a chat message while maintaining conversation history."""
    response = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    # Extract and return just the latest AI response
    return response["messages"][-1].content

# 4. Main Execution
"""
graph TD

    A[chat_with_memory] --> B[app.invoke]
    B --> C[process_messages]
    C --> D[create_augmented_response]
    D --> E[vector_store.search]
    D --> F[model.invoke]
    F --> G[Return Response]

    StateGraph: Manages the workflow structure
    vector_store: Holds document embeddings for semantic search
    model: The LLM (GPT-4) that generates responses
    MemorySaver: Handles conversation persistence
"""
if __name__ == "__main__":
    # Initialize Langtrace
    langtrace.init(api_key=os.environ['LANGTRACE_KEY'])  # Use the environment variable for the API key

    # Initialize OpenAI
    model = ChatOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        model="gpt-4"
    )
    
    # Check if pdf arg is provided
    if len(sys.argv) != 2:
        print("Usage: python PDF_agent.py <path_to_pdf>")
        sys.exit(1)

    # Setup vector store
    persist_directory = 'data/chroma/'
    reset_vector_store(persist_directory)
    
    # Take argument
    pdf_path = sys.argv[1]
    
    # Ensure file exists
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith('.pdf'):
        print(f"Invalid file: {pdf_path}. Please ensure it exists and is a PDF file.")
        sys.exit(1)

    docs, splits = process_documents([pdf_path])
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # View vector store contents for debugging
    # print("\nViewing Chroma Database Contents:")
    # view_chroma_database(vector_store)
    
    # Extract metadata
    if docs:
        metadata = extract_metadata(docs[0])
        print("\nDocument Metadata:")
        print(f"Date: {metadata['date'] or 'Not found'}")
        print(f"Author: {metadata['author'] or 'Not found'}")
        print("Number of pages:", len(docs),"\n")
    
    # Initialize chat application
    workflow = StateGraph(state_schema=MessagesState)
    app = setup_chat_workflow(vector_store)
    thread_id = "user_session_1"
    
    # Start chat loop
    try:
        while True:
            question = input("Enter a question (Or press Ctrl+C to quit): ")
            response = chat_with_memory(question, app, thread_id)
            print(f"\nAssistant: {response}\n")
    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")
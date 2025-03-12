import os
import sys
import numpy as np
import shutil
from typing import List
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Initialize the model
model = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    model="gpt-4"
)

# Function to reset the vector store
def reset_vector_store(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Vector store at '{persist_directory}' has been reset.")
    else:
        print(f"No existing vector store found at '{persist_directory}'.")

# LOAD PDF
loaders = [
    PyPDFLoader("MotivationletterSDKIO.pdf")
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# SPLITTER with new chunk sizes
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      # New chunk size
    chunk_overlap=50,    # New overlap
    separators=["\n\n", "\n", ".", "!", "?", ";", " "]
)

# Split the documents into smaller chunks
splits = text_splitter.split_documents(docs)

# VECTOR STORE
persist_directory = 'data/chroma/'

# Reset the vector store
reset_vector_store(persist_directory)

# Define embeddings before using it in vector store
embeddings = OpenAIEmbeddings()

# Reinitialize the vector store with new documents
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)

#print(vector_store._collection.count())

# Create a workflow with memory management
workflow = StateGraph(state_schema=MessagesState)

def create_augmented_response(state: MessagesState, vector_store):
    """Generate responses using both chat history and retrieved context"""
    # Get the latest question
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    
    # Retrieve relevant documents
    results = vector_store.max_marginal_relevance_search(
        query=latest_msg,
        k=3,
        fetch_k=6,
        lambda_mult=0.7
    )
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
    return {"messages": response}

def setup_chat_workflow(vector_store):
    """Set up the chat workflow with memory"""
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
    """Handle chat interaction with memory"""
    response = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    # Extract and return just the latest AI response
    return response["messages"][-1].content

def extract_metadata(doc):
    """Extract metadata using LLM to analyze the document content"""
    system_prompt = """You are a metadata extraction assistant. Given a document, extract the following metadata:
    1. Date (if present)
    2. Author (if present)
    Return ONLY a JSON-like string in this exact format:
    {"date": "found_date or null", "author": "found_author or null"}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract metadata from this text:\n\n{doc.page_content}")
    ]
    
    response = model.invoke(messages)
    
    try:
        # Clean up the response to get just the JSON string
        import json
        json_str = response.content.strip()
        if json_str.startswith("```") and json_str.endswith("```"):
            json_str = json_str[3:-3].strip()
        metadata = json.loads(json_str)
        return metadata
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return {"date": None, "author": None}

# Function to view the contents of the Chroma vector store
def view_chroma_database(vector_store):
    """Print the number of documents and their contents in the Chroma vector store."""
    count = vector_store._collection.count()
    print(f"Number of documents in the Chroma database: {count}")
    
    # Retrieve and print the documents using the get() method
    all_data = vector_store.get()  # Use the get() method to retrieve all documents
    for i in range(len(all_data["ids"])):
        doc_id = all_data["ids"][i]
        content = all_data["documents"][i]
        print(f"\nDocument ID: {doc_id}, \nContent: {content}")

# Example usage
if __name__ == "__main__":
    # Extract and print metadata from the first document
    if docs:
        metadata = extract_metadata(docs[0])
        print("\nDocument Metadata:")
        print(f"Date: {metadata['date'] or 'Not found'}")
        print(f"Author: {metadata['author'] or 'Not found'}\n")

    # View the Chroma database contents
    # view_chroma_database(vector_store)

    # Set up the chat application with memory
    app = setup_chat_workflow(vector_store)
    thread_id = "user_session_1"  # You can use different IDs for different chat sessions

    while True:
        question = input("Enter a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
            
        response = chat_with_memory(question, app, thread_id)
        print(f"\nAssistant: {response}\n")
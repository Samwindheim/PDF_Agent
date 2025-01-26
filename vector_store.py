import os
from openai import OpenAI
import sys
import numpy as np
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Update API key initialization
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# LOAD PDF
loaders = [
    PyPDFLoader("MotivationletterSCSSO.pdf")
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Increase chunk size to reduce number of splits
    chunk_overlap=50,    # Maintain some overlap for context
    separators=["\n\n", "\n", ".", "!", "?", ";"]  # Prioritize sentence-ending punctuation
)

def print_splits(splits, num_chars=200):
    """
    Print the content of each split with optional character limit.
    
    Args:
        splits: List of document splits
        num_chars: Number of characters to display from each split
    """
    print(f"Total number of splits: {len(splits)}\n")
    
    for i, split in enumerate(splits, 1):
        print(f"Split {i}:")
        print(f"Page: {split.metadata.get('page', 'unknown')}")
        print(f"Content: {split.page_content}")  # Display first num_chars characters
        print("-" * 80 + "\n")

splits = text_splitter.split_documents(docs)

# VECTOR STORE
persist_directory = 'data/chroma/'

# Define embeddings before using it in vector store
embeddings = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)

#print(vector_store._collection.count())

# Define a function to perform RAG
def retrieval_augmented_generation(question, k=3):
    """
    Perform Retrieval-Augmented Generation to answer a question.
    
    Args:
        question: The question to ask
        k: Number of relevant documents to retrieve
    """
    # Retrieve relevant documents
    results = vector_store.max_marginal_relevance_search(
        query=question,
        k=k,
        fetch_k=6,
        lambda_mult=0.7
    )
    
    # Extract content from results
    context = " ".join([doc.page_content for doc in results])
    
    # Manually construct messages
    formatted_messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant analyzing a motivation letter. 
            Use only the provided context to answer questions. 
            If the information isn't in the context, say 'I cannot find this information in the provided context.'
            Be concise and specific in your responses."""
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a clear and specific answer based on the context above."
        }
    ]
    
    # Generate an answer using the chat API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=formatted_messages,
        max_tokens=150,
        temperature=0.7
    )
    
    print(f"Question: {question}")
    print(f"Answer: {response.choices[0].message.content.strip()}\n")

# Example usage
if __name__ == "__main__":
    while True:
        question = input("Enter a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        retrieval_augmented_generation(question)
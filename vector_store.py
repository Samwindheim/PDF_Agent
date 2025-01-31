import os
import sys
import numpy as np
import shutil
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI as LangchainOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Update API key initialization
client = LangchainOpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Function to reset the vector store
def reset_vector_store(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Vector store at '{persist_directory}' has been reset.")
    else:
        print(f"No existing vector store found at '{persist_directory}'.")

# LOAD PDF
loaders = [
    PyPDFLoader("MotivationletterSCSSO.pdf")
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# SPLITTER with new chunk sizes
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,      # New chunk size
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

# Build prompt with clearer instructions
template = """Use the following pieces of context to answer the question at the end. 
Use three sentences maximum. Keep the answer as concise as possible.
Context: {context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Define a function to perform RAG using QA chain
def retrieval_augmented_generation(question, k=5):
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
    
    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=client,  # Use the Langchain-compatible OpenAI instance
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    result = qa_chain.invoke({"query": question, "context": context})
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}\n")

# Example usage
if __name__ == "__main__":
    while True:
        question = input("Enter a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        retrieval_augmented_generation(question)
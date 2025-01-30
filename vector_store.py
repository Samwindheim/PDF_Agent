import os
import sys
import numpy as np
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

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Define a function to perform RAG using QA chain
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
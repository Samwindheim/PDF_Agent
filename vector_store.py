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

# Update API key initialization
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# LOAD PDF
loaders = [
    PyPDFLoader('MotivationletterSCSSO.pdf')
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
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

# Embedding
embedding_1 = embeddings.embed_query("I love programming")
embedding_2 = embeddings.embed_query("I love music")
embedding_3 = embeddings.embed_query("What's the weather like in Konstanz?")

print(np.dot(embedding_1, embedding_2))
print(np.dot(embedding_1, embedding_3))
print(np.dot(embedding_2, embedding_3))













import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['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']

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# text splitter
text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,  
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

c_splitter = CharacterTextSplitter(
    chunk_size=26,
    chunk_overlap=4
)

split_text = r_splitter.split_text(text)

print(split_text)

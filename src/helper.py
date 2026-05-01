from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config import settings

# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

def load_pdf_files(data):
#   loader = PyPDFLoader(data)
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_page_content(docs: List[Document]) -> List[Document]:
  minimal_docs: List[Document] = []
  for doc in docs:
    src = doc.metadata.get("source")
    minimal_docs.append(Document(page_content=doc.page_content, metadata = {"source": src}))
  return minimal_docs

def text_splitter(minimal_docs):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
  text_chunk = text_splitter.split_documents(minimal_docs)
  return text_chunk


def download_embeddings():
  embeddings = HuggingFaceEmbeddings(
      model_name = settings.EMBEDDING_MODEL
  )
  return embeddings
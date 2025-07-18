from langchain_core.tools import tool
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@tool
def search1(query: str, file_path: str) -> str:
    """
    Tool to search content within a PDF using vector similarity.

    Arguments:
        query (str): The user query.
        file_path (str): The path to the PDF file.

    Returns:
        str: Matched context from the document.
    """

    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File at {file_path} not found.")
    if not file_path.endswith(".pdf"):
        raise ValueError("Please provide a valid PDF file.")

    
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading the PDF: {e}")

    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(pages)
    except Exception as e:
        raise RuntimeError(f"Could not split the document: {e}")

    
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
    except Exception as e:
        raise RuntimeError(f"Vectorization failed: {e}")

    if not vectorstore:
        raise ValueError("Vector store was not created.")

   
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)

    if not docs:
        raise ValueError("No matching document found in the vector store.")

    result = []
    for i, doc in enumerate(docs):
        result.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(result)

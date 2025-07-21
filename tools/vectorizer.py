import os
import tempfile
from typing import Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from supabase import create_client
from PyPDF2 import PdfReader





SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def read_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type.")

def upload_to_supabase(local_path: str, remote_path: str):
    with open(local_path, "rb") as f:
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(remote_path, f)
    return res

def vectorize_and_upload(file_path: str, document_id: str):
    content = read_file(file_path)
    chunks = splitter.split_text(content)
    if not chunks:
        raise ValueError("No text chunks found in the document. Is the file empty or unreadable?")
    docs = [Document(page_content=c) for c in chunks]
    vectordb = FAISS.from_documents(docs, embedding)

    with tempfile.TemporaryDirectory() as tmpdir:
        vectordb.save_local(tmpdir)
        upload_to_supabase(f"{tmpdir}/index.faiss", f"{document_id}/index.faiss")
        upload_to_supabase(f"{tmpdir}/index.pkl", f"{document_id}/index.pkl")

        return {
            "document_id": document_id,
            "supabase_path": f"{document_id}/",
        }

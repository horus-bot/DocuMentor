import os
import tempfile
from supabase import create_client, Client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from langchain_core.tools import tool
from dotenv import load_dotenv


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


BUCKET_NAME = "vectorstore"  # 
FAISS_FILES = ["index.faiss", "index.pkl"]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vectorstore_from_supabase(supabase_path: str) -> FAISS:
    if not supabase_path.endswith('/'):
        supabase_path += '/'
    with tempfile.TemporaryDirectory() as temp_dir:
        for fname in FAISS_FILES:
            full_path = f"{supabase_path}{fname}"
            res = supabase.storage.from_(SUPABASE_BUCKET).download(full_path)
            local_path = os.path.join(temp_dir, fname)
            with open(local_path, "wb") as f:
                f.write(res)
        return FAISS.load_local(temp_dir, embedding_model, allow_dangerous_deserialization=True)



class search_parameter(BaseModel):
    query:str
    supabase_path:str

@tool(args_schema=search_parameter)
def search1(parameter:search_parameter) -> str:
    """
    Tool to search content within a PDF using vector similarity.

    Always provide both parameters when calling this tool:
    - 'query' (str): The user's search question or keywords.
    - 'supabase_path' (str): The Supabase Storage folder path for the document's vector index files (e.g., the document ID or unique folder name). This is always available in your state as 'supabase_path'.

    Returns:
        str: Matched context from the document, or a message if no relevant information is found.
    """
    query = parameter.query
    supabase_path = parameter.supabase_path
    try:
        vectorstore = load_vectorstore_from_supabase(supabase_path)
        if not vectorstore:
            return "Sorry, I could not find the document or its vector index. Please check the document path."
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the document for your query."
        result = []
        for i, doc in enumerate(docs):
            result.append(f"Document {i + 1}:\n{doc.page_content}")

        if not result:
            raise ValueError("the vvector databsse is not found")  
        else :
            return "\n\n".join(result)  
        
    except Exception as e:
        return f"Error searching the document: {str(e)}"

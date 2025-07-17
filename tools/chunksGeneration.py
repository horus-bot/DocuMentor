from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from typing import List
from langchain_core.documents import Document


def chunks(path_dir:str)->List[Document]:
    """
    this is the pdf load with exception handling.
    Args:
        path_dir (str): Path to the PDF file.

    Returns:
        List[Document]: A list of document chunks.  
    """
    if not os.path.exists(path_dir):
        raise FileNotFoundError(f"file in {path_dir} not found")
    if not path_dir.endswith(".pdf"):
        raise ValueError("please input a valid pdf")
    
    try:
        loader = PyPDFLoader(path_dir)
        page=loader.load()
    except Exception as e:
        raise RuntimeError("error laoding the pdf")    
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " "]

        )
        chunks = splitter.split_documents(page)

    except Exception as e :
        raise RuntimeError("couldnt split the files") 

    return chunks

if __name__ == "__main__":
    print(chunks("test.pdf"))

     

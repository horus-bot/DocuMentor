from tools.chunksGeneration import chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def vectorization(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """ 
    This will be used to create vector database.
    
    Args:
        chunks (List[Document]): The chunks returned by your text splitter.
        model_name (str): HuggingFace embedding model to use. 

    Returns:
        FAISS: A FAISS vector store initialized with the embedded chunks.   
    """
    try:
        
        embedding = HuggingFaceEmbeddings(model_name=model_name)

        
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)

        
        return vectorstore 
     
    except Exception as e:
        
        raise RuntimeError("Can't do the vectorization of the chunks") 
     
if __name__ == "__main__":
    print(vectorization(chunks("test.pdf")))

from langchain_core.tools import tool
from langchain_community.vectorstores.faiss import FAISS

@tool
def search(query:str,vectorstore:FAISS)->str:
    """
    this is the tool for searching through the uploaded pdf this pdf 
    argiment:
            query : this is the query entered by the user 
            vectorstore: this is the vector db of the pdf 
    """
    reterive = vectorstore.as_retriever()
    docs=reterive.invoke(query)
    result=[]
    if not docs:
        raise ValueError("no document dound in vectorestore")

    for i ,docs in enumerate(docs):
        result.append(f"document {i+1}\n {docs.page_content} ")

    return "\n\n".join(result)
    
if __name__ == "__main__":
    from tools.chunksGeneration import chunks
    from tools.vectorConvertor import vectorization

    file_path = "test.pdf"
    chunked_docs = chunks(file_path)
    vs = vectorization(chunked_docs)

    query = "What is absorption?"
    print(search.invoke({"query": query, "vectorstore": vs}))



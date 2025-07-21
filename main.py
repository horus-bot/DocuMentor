from dotenv import load_dotenv
load_dotenv() 


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
import os
import tempfile  

from tools.vectorizer import vectorize_and_upload
from state import AgenState
from langgraph.graph import StateGraph,END,START
from agents import model,condition
from langgraph.prebuilt import ToolNode
from tools.search1 import search1
from langchain_core.messages import HumanMessage, BaseMessage
from agents.tool_adjust import take_action

app = FastAPI()
 

@app.post("/upload-doc/")
async def upload_doc(file: UploadFile = File(...)):
    try:
        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "txt"]:
            raise HTTPException(status_code=400, detail="Only .pdf and .txt supported.")

        doc_id = str(uuid4())
        tmpdir = tempfile.gettempdir()  
        file_path = os.path.join(tmpdir, f"{doc_id}.{ext}") 

        with open(file_path, "wb") as f:
            f.write(await file.read())

        result = vectorize_and_upload(file_path, doc_id)
        return JSONResponse(status_code=200, content={
            "status": "success",
            "document_id": result["document_id"],
            "supabase_path": result["supabase_path"]
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)  
        raise HTTPException(status_code=500, detail=str(e))






graph = StateGraph(AgenState)
graph.add_node("ai", model.ai)
graph.add_node("tool", take_action)


graph.set_entry_point("ai") 
graph.set_finish_point("ai")

    
graph.add_conditional_edges(
        "ai",
        condition.should_continue,
        {
            True: "ai",      
            False: END,     
            "tool": "tool",  
        }
    )

graph.add_edge("tool", "ai")

agent = graph.compile()

def serialize_message(msg):
    if isinstance(msg, BaseMessage):
        return {"type": msg.type, "content": msg.content}
    return str(msg)

@app.post("/ask-agent/")
async def ask_agent(question: str, document_id: str):
    try:
        state = {
            "messages": [HumanMessage(content=question)],
            "supabase_path": document_id  
        }
        result = agent.invoke(state)#github
        # Always serialize the full conversation
        if isinstance(result, dict) and "messages" in result:
            result["messages"] = [serialize_message(m) for m in result["messages"]]
        return JSONResponse(status_code=200, content={"result": result}),search1("what is the absorption","736c9cdb4-1900-4613-81d2-02d41f5746cf")
    except Exception as e:
        print("AGENT ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


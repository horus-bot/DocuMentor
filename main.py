from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
import os
import tempfile  

from tools.vectorizer import vectorize_and_upload

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

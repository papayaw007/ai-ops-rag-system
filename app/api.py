from fastapi import FastAPI, UploadFile, File
import os
import shutil

from app.rag import RAGEngine

app = FastAPI()

@app.get("/")
def home():
    return {"status": "working"}

rag = RAGEngine()


# ---------- Upload PDF ----------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = rag.load_pdf(file_path)

    return {
        "message": "PDF processed successfully",
        "chunks_created": num_chunks
    }


# ---------- Ask Question ----------
@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question")

    if not question:
        return {"error": "No question provided"}

    result = rag.ask(question)

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    }
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import ask_rag, get_collection_count


app = FastAPI(title="UChicago MS ADS RAG Backend")

# For local testing and Lovable. Later replace "*" with your Lovable domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    k: int = 3


@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "status": "running",
        "service": "UChicago MS ADS RAG Backend",
        "collection_count": get_collection_count(),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "collection_count": get_collection_count()}


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = ask_rag(question=question, show_sources=False, k=req.k)

    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
    }

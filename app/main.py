from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGAssistant


class QuestionRequest(BaseModel):
    """Request payload for the question endpoint."""

    question: str = Field(..., min_length=1, description="Question to answer using the handbook PDF")


app = FastAPI(
    title="School RAG Assistant API",
    description="FastAPI service that answers questions from a local PDF using Gemini and FAISS.",
    version="1.0.0",
)


@app.on_event("startup")
def initialize_rag_pipeline() -> None:
    """Load the PDF and build the FAISS vector index when the server starts."""
    try:
        pdf_path = "data/handbook.pdf"
        app.state.rag_assistant = RAGAssistant(pdf_path=pdf_path)
        app.state.rag_assistant.initialize_pipeline()
    except FileNotFoundError as exc:
        raise RuntimeError(f"PDF file not found: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Missing API configuration: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"RAG initialization failed: {exc}") from exc


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Check whether the API is running."""
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: QuestionRequest) -> Dict[str, str]:
    """Answer a question using the initialized assistant."""
    try:
        if not hasattr(app.state, "rag_assistant"):
            raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")

        result = app.state.rag_assistant.ask_question(payload.question)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Assistant returned an invalid response format.")

        answer = result.get("answer")
        source = result.get("source")

        if not isinstance(answer, str) or not isinstance(source, str):
            raise HTTPException(status_code=500, detail="Assistant response is missing required fields.")

        return {"answer": answer, "source": source}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Server error while processing the question: {exc}") from exc

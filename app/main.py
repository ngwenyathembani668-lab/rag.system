from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.rag_pipeline import RAGAssistant

logger = logging.getLogger(__name__)


def load_environment() -> None:
    """Load environment variables from the project root regardless of cwd depth."""
    candidate_dirs = []

    cwd = Path.cwd().resolve()
    current_file_parent = Path(__file__).resolve().parent.parent

    candidate_dirs.extend(
        [
            cwd,
            cwd.parent if cwd.parent != cwd else None,
            current_file_parent,
            current_file_parent.parent if current_file_parent.parent != current_file_parent else None,
        ]
    )

    seen: set[str] = set()
    for directory in candidate_dirs:
        if directory is None:
            continue
        directory_str = str(directory)
        if directory_str in seen:
            continue
        seen.add(directory_str)

        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.info("Loaded environment variables from %s", env_path)
            return

    load_dotenv(override=False)


load_environment()


class QuestionRequest(BaseModel):
    """Strict request payload for the question endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer using the handbook PDF and ZAIO website knowledge base.",
    )


def initialize_rag_assistant() -> RAGAssistant:
    """Create and initialize the multi-source RAG assistant at startup."""
    assistant = RAGAssistant(
        pdf_path="data/handbook.pdf",
        website_url="https://zaio.io",
    )
    assistant.initialize_pipeline()
    return assistant


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the assistant during startup without crashing the process."""
    app.state.rag_assistant: Optional[RAGAssistant] = None

    try:
        app.state.rag_assistant = initialize_rag_assistant()
        logger.info("RAG assistant initialized successfully during startup.")
    except FileNotFoundError as exc:
        logger.warning("Handbook PDF missing at startup: %s", exc)
    except ValueError as exc:
        logger.warning("Startup configuration error: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive production guard
        logger.exception("Unexpected startup error while initializing the RAG assistant: %s", exc)

    yield


app = FastAPI(
    title="School RAG Assistant API",
    description="FastAPI service that answers questions from the handbook PDF and ZAIO website using Gemini and FAISS.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, object]:
    """Provide a lightweight monitoring endpoint for orchestrators and automation tools."""
    initialized = getattr(app.state, "rag_assistant", None) is not None
    return {"status": "ok", "initialized": initialized}


@app.post("/ask")
async def ask_question(payload: QuestionRequest) -> Dict[str, str]:
    """Answer a question using the configured multi-source assistant."""
    rag_assistant = getattr(app.state, "rag_assistant", None)
    if rag_assistant is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Please verify the handbook PDF and API configuration.",
        )

    try:
        result = rag_assistant.ask_question(payload.question)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Assistant returned an invalid response format.")

        answer = result.get("answer")
        source = result.get("source")
        if not isinstance(answer, str) or not isinstance(source, str):
            raise HTTPException(status_code=500, detail="Assistant response is missing required fields.")

        return {"answer": answer, "source": source}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled exception while processing /ask request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Server error while processing the question: {exc}",
        ) from exc

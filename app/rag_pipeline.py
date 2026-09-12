import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


class RAGAssistant:
    def __init__(self, pdf_path: str):
        """Initialize the local embedding model, Gemini LLM, and project PDF path."""
        load_dotenv()

        self.pdf_path = pdf_path
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if os.path.isabs(self.pdf_path):
            self.resolved_pdf_path = self.pdf_path
        else:
            self.resolved_pdf_path = os.path.join(self.project_root, self.pdf_path)

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Gemini API key. Add GEMINI_API_KEY or GOOGLE_API_KEY to your .env file."
            )

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=self.api_key,
        )
        self.vectorstore = None

    def initialize_pipeline(self):
        """Load the PDF, split the text into chunks, and build a local FAISS index."""
        default_pdf = os.path.join(self.project_root, "data", "handbook.pdf")

        if os.path.exists(self.resolved_pdf_path):
            pdf_path = self.resolved_pdf_path
        elif os.path.exists(default_pdf):
            pdf_path = default_pdf
        else:
            raise FileNotFoundError(
                f"Could not find the PDF at '{self.resolved_pdf_path}' or '{default_pdf}'."
            )

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            metadata = chunk.metadata or {}
            metadata["page"] = int(metadata.get("page", 0))
            chunk.metadata = metadata

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return self.vectorstore

    def _extract_page_number(self, document: Document) -> int:
        """Extract the page number from metadata and convert from zero-based PDF indexing."""
        metadata = getattr(document, "metadata", {}) or {}
        page_number = int(metadata.get("page", 0)) + 1
        return page_number

    def _normalize_answer(self, answer: str) -> str:
        """Normalize model output and enforce exact fallback behavior for unknown answers."""
        cleaned = re.sub(r"\s+", " ", answer or "").strip()
        cleaned = cleaned.strip("\"'` ")

        if not cleaned:
            return "I don't know."

        lower = cleaned.lower()
        if "i don't know" in lower or "i do not know" in lower:
            return "I don't know."

        return cleaned

    def ask_question(self, question: str) -> dict:
        """Find the best matches in the vectorstore and answer using Gemini."""
        if self.vectorstore is None:
            self.initialize_pipeline()

        relevant_documents: List[Document] = self.vectorstore.similarity_search(question, k=3)

        if not relevant_documents:
            return {"answer": "I don't know.", "source": "N/A"}

        context_parts = []
        for doc in relevant_documents:
            page_number = self._extract_page_number(doc)
            context_parts.append(f"[Page {page_number}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        system_message = SystemMessage(
            content=(
                "You are a strict answer-only assistant. Use only the provided context. "
                "If the requested information is not explicitly present in the context, respond with exactly 'I don't know.'. "
                "Do not guess, do not speculate, and do not provide answers outside the context. "
                "Return only the final answer text without markdown, quotes, or extra commentary."
            )
        )

        user_message = HumanMessage(
            content=(
                f"Question: {question}\n\nContext:\n{context}"
            )
        )

        response = self.llm.invoke([system_message, user_message])
        answer = response.content if hasattr(response, "content") else str(response)
        normalized_answer = self._normalize_answer(answer)

        if normalized_answer == "I don't know.":
            return {"answer": "I don't know.", "source": "N/A"}

        source_page = self._extract_page_number(relevant_documents[0])
        return {"answer": normalized_answer, "source": f"Page {source_page}"}

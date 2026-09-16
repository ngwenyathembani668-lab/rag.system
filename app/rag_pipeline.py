import os
import re
from typing import List
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class RAGAssistant:
    def __init__(self, pdf_path: str, website_url: str = "https://zaio.io"):
        """Initialize the Cloud Embeddings model, Gemini LLM, and path definitions."""
        load_dotenv()

        self.pdf_path = pdf_path
        self.target_url = website_url
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

        # CLOUD EMBEDDINGS FIX: Bypasses local binary blocks by shifting execution to Google's API
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-005", 
            google_api_key=self.api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            api_key=self.api_key,
        )
        self.vectorstore = None

    def _resolve_pdf_path(self) -> str:
        """Find the PDF on disk, accepting standard naming patterns."""
        candidates = [
            self.resolved_pdf_path,
            os.path.join(self.project_root, "data", "handbook.pdf"),
            os.path.join(self.project_root, "data", "handbook.pdf.pdf"),
        ]

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Could not find the PDF at '{self.resolved_pdf_path}' or in the project data folder."
        )

    def _scrape_and_clean_website(self) -> List[Document]:
        """Part 1: Crawl, extract, and clean text from the ZAIO website."""
        try:
            response = requests.get(self.target_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove navigation elements, headers, footers, scripts, and styling layers
            for element in soup(["nav", "header", "footer", "script", "style"]):
                element.decompose()
                
            clean_text = soup.get_text(separator="\n")
            clean_text = re.sub(r"\n+", "\n", clean_text).strip()
            
            return [Document(page_content=clean_text, metadata={"source": "Website", "url": self.target_url})]
        except Exception as e:
            print(f"⚠️ Warning: Web scraping failed ({e}). Continuing with PDF only.")
            return []

    def initialize_pipeline(self):
        """Load the PDF, scrape the website, chunk the text, and build a unified FAISS index."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        all_documents = []

        # 1. Load Handbook PDF
        try:
            pdf_path = self._resolve_pdf_path()
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                page_num = int(doc.metadata.get("page", 0)) + 1
                doc.metadata = {"source": "Handbook", "page": page_num}
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"⚠️ PDF Load Warning: {e}")

        # 2. Scrape ZAIO Website
        web_docs = self._scrape_and_clean_website()
        all_documents.extend(web_docs)

        if not all_documents:
            raise ValueError("No data sources were loaded successfully.")

        # 3. Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(all_documents)

        # 4. Generate cloud embeddings and build the unified FAISS database
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return self.vectorstore

    def _extract_page_number(self, document: Document) -> int:
        """Extract page number from metadata and translate zero-based PDF indexing to one-based."""
        metadata = getattr(document, "metadata", {}) or {}
        page_number = int(metadata.get("page", 0))
        return page_number

    def _normalize_answer(self, answer: str) -> str:
        """Normalize model output and rigidly enforce the fallback behavior."""
        if isinstance(answer, list):
            text_parts = [item.get("text", str(item)) if isinstance(item, dict) else getattr(item, "text", str(item)) for item in answer]
            answer = "".join(text_parts)
            
        cleaned = re.sub(r"\s+", " ", str(answer) or "").strip().strip("\"'` ")
        lower = cleaned.lower()
        fallback_msg = "I could not find that information in the available knowledge base."

        if not cleaned or "don't know" in lower or "not find" in lower or "not present" in lower or "available knowledge" in lower:
            return fallback_msg

        return cleaned

    def ask_question(self, question: str) -> dict:
        """Find the best matches in the vector store and answer using Gemini."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.vectorstore is None:
            self.initialize_pipeline()

        relevant_documents: List[Document] = self.vectorstore.similarity_search(question, k=3)
        fallback_msg = "I could not find that information in the available knowledge base."

        if not relevant_documents:
            return {"answer": fallback_msg, "source": "N/A"}

        context_parts = []
        for doc in relevant_documents:
            src_type = doc.metadata.get("source", "Unknown")
            if src_type == "Handbook":
                ref = f"[Source: Handbook - Page {doc.metadata.get('page')}]"
            else:
                ref = f"[Source: Website - URL {doc.metadata.get('url')}]"
            context_parts.append(f"{ref}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        system_message = SystemMessage(
            content=(
                "You are a strict answer-only assistant. Use only the provided context. "
                "If the requested information is not explicitly present in the context, respond with exactly "
                "'I could not find that information in the available knowledge base.'. "
                "Do not guess, do not speculate, and do not provide answers outside the context. "
                "Return only the final answer text without markdown, quotes, or extra commentary."
            )
        )

        user_message = HumanMessage(content=f"Question: {question}\n\nContext:\n{context}")

        response = self.llm.invoke([system_message, user_message])
        answer = response.content if hasattr(response, "content") else str(response)
        normalized_answer = self._normalize_answer(answer)

        if normalized_answer == fallback_msg:
            return {"answer": fallback_msg, "source": "N/A"}

        # Part 3: Dynamic validation source citation formatting engine
        primary_doc = relevant_documents[0]
        if primary_doc.metadata.get("source") == "Handbook":
            source_citation = f"Student Handbook - Page {primary_doc.metadata.get('page')}"
        else:
            source_citation = str(primary_doc.metadata.get("url", self.target_url))

        return {"answer": normalized_answer, "source": source_citation}
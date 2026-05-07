"""
app/retriever.py
────────────────
ChromaDB vector store with flexible embeddings (Ollama, OpenAI, Google, etc.).
"""

from __future__ import annotations

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

from app.config import settings


# ── Singleton embedding model ────────────────────────────────────────────────

_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Return cached embeddings instance based on configured provider."""
    global _embeddings
    if _embeddings is None:
        provider = settings.llm_provider.lower()
        
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not set for embeddings.")
            _embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model="text-embedding-3-small",  # cheaper embedding model
            )
        
        elif provider == "google":
            if not settings.google_api_key:
                raise ValueError("Google API key not set for embeddings.")
            _embeddings = GoogleGenerativeAIEmbeddings(
                api_key=settings.google_api_key,
                model="models/embedding-001",
            )
        
        else:  # Default to Ollama
            _embeddings = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )
    return _embeddings


# ── Vector store ─────────────────────────────────────────────────────────────

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _vectorstore


def get_retriever():
    """Return a LangChain retriever that fetches top-k chunks."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )


def add_documents(docs: List[Document]) -> int:
    """Embed and persist a list of Document objects. Returns count added."""
    vs = get_vectorstore()
    vs.add_documents(docs)


    return len(docs)


def collection_count() -> int:
    """Return total number of chunks stored in ChromaDB."""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0


def reset_collection() -> None:
    """Delete all documents from the collection and reset cache."""
    global _vectorstore
    _vectorstore = None

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    # Try hard delete first (fast and complete)
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass

    # Always ensure an empty collection exists after reset
    collection = client.get_or_create_collection(settings.chroma_collection_name)
    try:
        collection.delete(where={})
    except Exception:
        # If clear-by-filter is unsupported, ignore — collection is already recreated.
        pass

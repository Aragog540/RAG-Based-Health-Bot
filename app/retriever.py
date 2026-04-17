"""
app/retriever.py
────────────────
ChromaDB vector store with Ollama embeddings.
Completely free — no external API calls.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List

from app.config import settings


# ── Singleton embedding model ────────────────────────────────────────────────

_embeddings: OllamaEmbeddings | None = None


def get_embeddings() -> OllamaEmbeddings:
    """Return cached OllamaEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
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
    """Delete all documents from the collection."""
    global _vectorstore
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    client.delete_collection(settings.chroma_collection_name)
    _vectorstore = None  # force re-creation on next call

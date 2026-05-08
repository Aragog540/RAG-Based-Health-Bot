"""
app/retriever.py
────────────────
ChromaDB vector store with flexible embeddings (Ollama, OpenAI, Google, etc.).
"""

from __future__ import annotations

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List

from app.config import settings


# ── Singleton embedding model ────────────────────────────────────────────────

_embeddings: OllamaEmbeddings | None = None


def get_embeddings() -> OllamaEmbeddings:
    """Return cached Ollama embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    return _embeddings


def get_retriever():
    """Return a simple ChromaDB-based retriever for local Ollama runs."""

    class ChromaRetriever:
        def invoke(self, query: str) -> List[Document]:
            collection = get_vectorstore()
            embeddings = get_embeddings()
            try:
                query_embedding = embeddings.embed_query(query)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=settings.retrieval_top_k,
                )
            except Exception:
                results = collection.get(limit=settings.retrieval_top_k)

            docs = []
            if results and results.get("documents"):
                for doc_text, metadata in zip(results["documents"][0], results.get("metadatas", [[]])[0]):
                    docs.append(Document(page_content=doc_text, metadata=metadata or {}))
            return docs

    return ChromaRetriever()


def add_documents(docs: List[Document]) -> int:
    """Embed and persist a list of Document objects. Returns count added."""
    collection = get_vectorstore()
    embeddings = get_embeddings()

    for i, doc in enumerate(docs):
        doc_id = f"doc_{i}_{hash(doc.page_content) % 10000}"
        try:
            embedding = embeddings.embed_query(doc.page_content)
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc.page_content],
                metadatas=[doc.metadata or {}],
            )
        except Exception:
            collection.add(
                ids=[doc_id],
                documents=[doc.page_content],
                metadatas=[doc.metadata or {}],
            )

    return len(docs)


def collection_count() -> int:
    """Return total number of chunks stored in ChromaDB."""
    try:
        collection = get_vectorstore()
        result = collection.get()
        return len(result.get("ids", [])) if result else 0
    except Exception:
        return 0


def reset_collection() -> None:
    """Delete all documents from the collection and reset cache."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass

    client.get_or_create_collection(
        settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_vectorstore():
    """Get or create the Chroma collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

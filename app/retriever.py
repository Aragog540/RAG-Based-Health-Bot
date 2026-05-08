"""
app/retriever.py
────────────────
ChromaDB vector store with Google embeddings.
"""

from __future__ import annotations

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List

from app.config import settings


# ── Singleton embedding model ────────────────────────────────────────────────

_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """Return cached Google embeddings instance."""
    global _embeddings
    if _embeddings is None:
        if not settings.google_api_key:
            raise ValueError("Google API key not set for embeddings.")
        _embeddings = GoogleGenerativeAIEmbeddings(
            api_key=settings.google_api_key,
            model="models/text-embedding-004",
        )
    return _embeddings


# ── Vector store ─────────────────────────────────────────────────────────────

_client: chromadb.Client | None = None


def get_chromadb_client() -> chromadb.Client:
    """Get ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_vectorstore():
    """Get or create Chroma collection."""
    client = get_chromadb_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def get_retriever():
    """Return a simple retriever that queries Chroma by semantic similarity."""
    
    class ChromaRetriever:
        def invoke(self, query: str) -> List[Document]:
            """Retrieve documents similar to the query."""
            embeddings = get_embeddings()
            query_embedding = embeddings.embed_query(query)
            
            collection = get_vectorstore()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=settings.retrieval_top_k,
            )
            
            docs = []
            if results and results.get("documents"):
                for doc_text, metadata in zip(results["documents"][0], results.get("metadatas", [[]])[0]):
                    docs.append(Document(page_content=doc_text, metadata=metadata or {}))
            return docs
    
    return ChromaRetriever()


def add_documents(docs: List[Document]) -> int:
    """Embed and persist a list of Document objects. Returns count added."""
    embeddings = get_embeddings()
    collection = get_vectorstore()
    
    for i, doc in enumerate(docs):
        embedding = embeddings.embed_query(doc.page_content)
        collection.add(
            ids=[f"doc_{i}_{hash(doc.page_content) % 10000}"],
            embeddings=[embedding],
            documents=[doc.page_content],
            metadatas=[doc.metadata or {}],
        )
    
    return len(docs)


def collection_count() -> int:
    """Return total number of chunks stored in ChromaDB."""
    try:
        collection = get_vectorstore()
        return collection.count()
    except Exception:
        return 0


def reset_collection() -> None:
    """Delete all documents from the collection and reset cache."""
    global _client
    client = get_chromadb_client()
    
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

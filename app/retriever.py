"""
app/retriever.py
────────────────
ChromaDB vector store with Google embeddings (falls back to default if API unavailable).
"""

from __future__ import annotations

import chromadb
from langchain_core.documents import Document
from typing import List

from app.config import settings


# ── Singleton embedding model ────────────────────────────────────────────────

_embeddings = None


def get_embeddings():
    """Return cached embedding function for ChromaDB."""
    global _embeddings
    if _embeddings is None:
        if settings.google_api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                _embeddings = GoogleGenerativeAIEmbeddings(
                    api_key=settings.google_api_key,
                    model="embedding-001",
                )
            except Exception as e:
                print(f"Google embeddings failed ({e}), using ChromaDB defaults")
                _embeddings = None  # Will use chromadb default
        else:
            print("No Google API key, using ChromaDB default embeddings")
            _embeddings = None
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
            collection = get_vectorstore()
            embeddings = get_embeddings()
            
            # If embeddings available, get query embedding; otherwise use text search
            if embeddings and hasattr(embeddings, 'embed_query'):
                try:
                    query_embedding = embeddings.embed_query(query)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=settings.retrieval_top_k,
                    )
                except Exception:
                    # Fallback to where search
                    results = collection.get(limit=settings.retrieval_top_k)
            else:
                # Use chromadb's built-in embedding (no API needed)
                results = collection.query(
                    query_texts=[query],
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
    collection = get_vectorstore()
    embeddings = get_embeddings()
    
    for i, doc in enumerate(docs):
        doc_id = f"doc_{i}_{hash(doc.page_content) % 10000}"
        
        # Try to use embeddings if available; otherwise let chromadb handle it
        if embeddings and hasattr(embeddings, 'embed_query'):
            try:
                embedding = embeddings.embed_query(doc.page_content)
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata or {}],
                )
            except Exception:
                # Fallback: add without embedding, let chromadb compute
                collection.add(
                    ids=[doc_id],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata or {}],
                )
        else:
            # Let chromadb compute embedding using its default
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
        # ChromaDB collections don't have .count(), get data and count
        result = collection.get()
        return len(result.get("ids", [])) if result else 0
    except Exception as e:
        print(f"Error counting collection: {e}")
        return 0


def reset_collection() -> None:
    """Delete all documents from the collection and reset cache."""
    global _client
    try:
        client = get_chromadb_client()
        client.delete_collection(settings.chroma_collection_name)
        # Recreate empty collection
        client.get_or_create_collection(settings.chroma_collection_name)
    except Exception as e:
        print(f"Error resetting collection: {e}")

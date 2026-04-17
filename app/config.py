"""
app/config.py
─────────────
All tuneable settings in one place.
No API keys needed — everything runs locally via Ollama.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── Ollama ──────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # Change to "mistral" or "phi3" if you have less RAM
    llm_model: str = "llama3"

    # Free embedding model — pull with: ollama pull nomic-embed-text
    embedding_model: str = "nomic-embed-text"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "medical_book"
    
    # ── RAG tuning ──────────────────────────────────────────
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # overlap between chunks
    retrieval_top_k: int = 5       # how many chunks to retrieve

    # ── App ─────────────────────────────────────────────────
    app_title: str = "Medical RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = False

settings = Settings()

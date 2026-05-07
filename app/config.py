"""
app/config.py
─────────────
All tuneable settings in one place.
No API keys needed — everything runs locally via Ollama.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── LLM Provider (ollama, openai, anthropic, etc.) ──────
    llm_provider: str = "ollama"  # Set via LLM_PROVIDER env var
    
    # ── Ollama Config ────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"  # Change to "mistral" or "phi3" if needed
    embedding_model: str = "nomic-embed-text"
    
    # ── OpenAI Config (if using OpenAI) ─────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    
    # ── Anthropic Config (if using Claude) ──────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # ── Google Config (if using Gemini) ────────────────────
    google_api_key: str = ""
    google_model: str = "gemini-1.5-flash"

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

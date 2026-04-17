# Medical RAG Chatbot (LangGraph + FastAPI + Ollama + ChromaDB)

A local-first Retrieval-Augmented Generation (RAG) chatbot for medical PDFs.

This project uses:
- FastAPI for the HTTP API
- LangGraph for orchestration
- ChromaDB as the vector store
- Ollama for local LLM and embeddings

No cloud API keys are required.

## Features

- Local medical Q and A over your PDF content
- Ingestion pipeline: PDF -> text -> chunks -> embeddings -> ChromaDB
- Agentic RAG graph with:
  - retrieval
  - relevance grading
  - answer generation
  - grounding check
- REST API with Swagger docs
- Health and source inspection endpoints
- Reset endpoint to clear indexed data
- Basic API tests with pytest

## Tech Stack

- Python 3.11 (recommended)
- FastAPI
- LangChain + LangGraph
- LangChain-Ollama
- LangChain-Chroma
- ChromaDB
- pypdf

## Project Structure

```text
RAG-Based-Health-Bot-main/
|- app/
|  |- config.py
|  |- graph.py
|  |- main.py
|  |- models.py
|  `- retriever.py
|- data/
|- scripts/
|  `- ingest.py
|- tests/
|  `- test_api.py
|- requirements.txt
`- README.md
```

## Prerequisites

1. Install Python 3.11
2. Install Ollama: https://ollama.com/download
3. Pull required models:

```powershell
ollama pull llama3
ollama pull nomic-embed-text
```

If Ollama is not running automatically:

```powershell
ollama serve
```

## Installation

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r requirements.txt
```

## Quick Start

1. Put your PDF in the data folder (example: data/Medical_book.pdf)
2. Ingest the PDF:

```powershell
py -3.11 scripts\ingest.py --pdf data\Medical_book.pdf
```

3. Start the API server:

```powershell
py -3.11 -m uvicorn app.main:app --reload --port 8000
```

4. Open API docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Service status, model info, chunk count |
| POST | /chat | Ask a question against indexed documents |
| POST | /ingest | Upload a PDF and ingest via API |
| GET | /sources | Preview indexed chunks |
| DELETE | /reset | Clear all indexed documents |

Note: the root endpoint / is not defined, so a 404 there is expected.

## Example Requests

Health:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health |
  Select-Object -ExpandProperty Content
```

Chat:

```powershell
$body = @{ question = "What are the symptoms of Type 2 Diabetes?" } | ConvertTo-Json
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/chat -Method POST -ContentType "application/json" -Body $body | Select-Object -ExpandProperty Content
```

Ingest via API:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/ingest -Method POST -Form @{ file = Get-Item "data\\Medical_book.pdf" }
```

## Configuration

Main settings are in app/config.py:

- ollama_base_url
- llm_model (default: llama3)
- embedding_model (default: nomic-embed-text)
- chroma_persist_dir
- chroma_collection_name
- chunk_size
- chunk_overlap
- retrieval_top_k

You can override these with environment variables or a .env file.

Example .env:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=medical_book
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
```

## Running Tests

```powershell
py -3.11 -m pytest -q
```

Current test layout includes:
- health endpoint checks
- basic chat request validation
- sources endpoint check
- optional skipped tests for destructive or model-dependent paths

## Architecture Overview

RAG flow in app/graph.py:

1. retrieve: get top-k chunks from ChromaDB
2. grade_documents: filter chunks by relevance
3. generate: produce answer from context
4. check_hallucination: verify grounding against source docs

The API returns:
- answer text
- source snippets with metadata
- grounded boolean

## Troubleshooting

1. Error: No module named uvicorn or fastapi
- Use the same interpreter for install and run (Python 3.11 commands above).

2. Error: No module named langchain_chroma
- Ensure requirements are installed from the updated requirements.txt.

3. /chat returns no documents indexed
- Run ingestion first with scripts/ingest.py.

4. Ollama-related failures
- Confirm Ollama is running and both models are pulled.

5. Editor shows unresolved imports but terminal run works
- VS Code may be pointed to a different interpreter.
- Select your Python 3.11 environment in VS Code.

## Recommended Python Version

This project is currently validated with Python 3.11.

Python 3.13 may fail with some pinned package builds in this dependency set.

## License

Add your preferred license here (for example: MIT).

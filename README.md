# 🏥 Medical RAG Chatbot — LangGraph + FastAPI

A medical book chatbot using LangGraph for agentic RAG,
FastAPI for the backend, ChromaDB as the vector store, and
**Ollama** (runs local LLMs — no API keys needed).

---

## 🗂 Project Structure

```
medical-rag-chatbot/
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── graph.py           # LangGraph RAG graph
│   ├── retriever.py       # ChromaDB vector retriever
│   ├── models.py          # Pydantic request/response schemas
│   └── config.py          # All settings in one place
├── scripts/
│   └── ingest.py          # PDF → chunk → embed → ChromaDB
├── data/
│   └── (put your PDF here)
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

### 1. Install Ollama (free, local LLM)

Windows → download from https://ollama.com/download


### 2. Pull a free model (choose one)
```bash
ollama pull llama3          # 4.7 GB — best quality
ollama pull mistral         # 4.1 GB — fast & good
ollama pull phi3            # 2.3 GB — lightest option
```

---

## 🚀 Quick Start

```bash
# 1. Go to project folder
cd medical-rag-chatbot

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment

# CMD:
venv\Scripts\activate

# PowerShell:
venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place your medical PDF in data folder
copy C:\path\to\your\medical_book.pdf data\

# 6. Ingest PDF (creates embeddings in ChromaDB)
python scripts\ingest.py --pdf data\medical_book.pdf

# 7. Make sure Ollama is installed and running
ollama --version

# (Optional – only if not running automatically)
ollama serve

# 8. Pull model (only first time)
ollama pull llama3

# 9. Run FastAPI server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** to use the Swagger UI.

---

## 💬 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a question, get an answer |
| GET | `/health` | Health check |
| GET | `/sources` | List indexed document chunks |
| DELETE | `/reset` | Clear vector store |

### Example cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of Type 2 Diabetes?"}'
```

---

## 🧠 Architecture

```
User Question
     │
     ▼
┌──────────────────────────────────────┐
│           LangGraph Graph            │
│                                      │
│  [retrieve_node]                     │
│       ↓ top-k chunks from ChromaDB   │
│  [grade_node]  ← checks relevance    │
│       ↓ relevant? yes/no             │
│  [generate_node]                     │
│       ↓ Ollama LLM answers           │
│  [hallucination_check_node]          │
│       ↓ grounded in docs?            │
│  [final_answer_node]                 │
└──────────────────────────────────────┘
     │
     ▼
  Answer + Sources
```

---

## 🔧 Configuration

Edit `app/config.py` to change:
- LLM model name (default: `llama3`)
- Embedding model (default: `nomic-embed-text` via Ollama)
- Chunk size / overlap
- Number of retrieved documents (top-k)
- ChromaDB persist path

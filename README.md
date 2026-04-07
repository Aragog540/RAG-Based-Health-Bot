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

# Windows → download from https://ollama.com/download


### 2. Pull a free model (choose one)
```bash
ollama pull llama3          # 4.7 GB — best quality
ollama pull mistral         # 4.1 GB — fast & good
ollama pull phi3            # 2.3 GB — lightest option
```

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project, then:
cd medical-rag-chatbot

# 2. Create virtual environment
python -m venv venv
Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your medical PDF in data/
copy C:\path\to\your\medical_book.pdf data\

# 5. Ingest PDF (chunks it, embeds it, stores in ChromaDB)
python scripts/ingest.py --pdf data/medical_book.pdf

# 6. Start Ollama in a separate terminal
ollama serve

# 7. Run the FastAPI server
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

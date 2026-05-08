# 🚀 Deployment Guide

## **Option 1: Deploy with OpenAI (Easiest for Global)**

### Step 1: Get OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy it

### Step 2: Create `.env` file
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### Step 3: Deploy to Render (Free Tier Available)
1. Push your code to GitHub
2. Go to https://render.com
3. Click "New +"  →  "Web Service"
4. Connect your GitHub repo
5. Set **Environment Variables**:
   - `LLM_PROVIDER`: `openai`
   - `OPENAI_API_KEY`: Your API key
6. Deploy!

---

## **Option 2: Deploy with Anthropic/Claude**

### Step 1: Get API Key
- Go to https://console.anthropic.com
- Create API key

### Step 2: Create `.env`
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

### Step 3: Deploy (same as above with Render)

---

## **Option 3: Deploy with Google Gemini (Cheapest)**

### Step 1: Get API Key
- Go to https://makersuite.google.com/app/apikey
- Create API key (works with free tier!)

### Step 2: Create `.env`
```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=your-key-here
GOOGLE_MODEL=gemini-2.0-flash
```

### Step 3: Deploy to Render

---

## **Popular Free/Cheap Deployment Platforms**

| Platform | Cost | GPU Support | Good For |
|----------|------|-------------|----------|
| **Render** | $0-7/mo | No (free tier) | FastAPI apps |
| **Railway** | Pay as you go | Optional | FastAPI apps |
| **Vercel** | Free serverless | No | Frontend only |
| **Google Cloud Run** | ~$0.50/mo | Optional | Containerized apps |
| **AWS Lambda** | Free tier | No | Serverless |
| **DigitalOcean** | $4-12/mo | Optional | Full VPS |

---

## **How to Deploy on Render (Step-by-Step)**

### 1. Create `requirements.txt` (make sure it includes new dependencies)
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
langchain>=0.4.0
langchain-community>=0.4.0
langchain-openai>=0.1.0  # Add this
langchain-anthropic>=0.1.0  # Add this
langchain-google-genai>=0.1.0  # Add this
chromadb==0.5.23
pypdf==5.0.1
python-dotenv==1.0.1
```

### 2. Create `Procfile` in root
```
web: cd RAG-Based-Health-Bot-main && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Push to GitHub
```bash
git add .
git commit -m "Add cloud LLM support + deployment"
git push
```

### 4. On Render.com
- New Web Service → Connect GitHub repo
- Set **Environment Variables**:
  ```
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-xxx
  PYTHON_VERSION=3.11
  ```
- Deploy!

---

## **Cost Breakdown** (Monthly Usage)

| Provider | Model | Cost | Example (1000 requests) |
|----------|-------|------|------------------------|
| **OpenAI** | gpt-3.5-turbo | $0.50/1M tokens | ~$2-5 |
| **Anthropic** | Claude 3 Sonnet | $3/1M in, $15/1M out | ~$3-8 |
| **Google** | Gemini 1.5 Flash | $0.075/1M tokens | ~$0.50-2 |
| **Ollama** | Local (free) | $0 | $0 (but no server) |

---

## **Next Steps**

1. **Add the new LLM packages** to requirements.txt:
   ```bash
   pip install langchain-openai langchain-anthropic langchain-google-genai
   pip freeze > requirements.txt
   ```

2. **Create `.env` file** locally with your chosen provider

3. **Test locally**:
   ```bash
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=sk-xxx
   python -m uvicorn app.main:app --port 8000
   ```

4. **Push to GitHub** → Deploy to Render/Railway

---

## **To Switch Providers Later**

Just change the `.env` variable:
```bash
LLM_PROVIDER=anthropic  # Switches to Claude
```

The code automatically uses the right LLM!

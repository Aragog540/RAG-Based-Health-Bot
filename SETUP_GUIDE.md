# 🚀 Setup Guide: Local + Global Deployment

## **Part 1: Local Development (Using Ollama - FREE)**

### ✅ You Already Have Everything!

**Current setup:**
- Ollama running locally ✅
- Medical_book.pdf ingested ✅
- Server running on localhost:8000 ✅

**To run locally:**
```bash
# Terminal 1: Make sure Ollama is running
ollama serve

# Terminal 2: Run the bot with Ollama
cd RAG-Based-Health-Bot-main
python -m uvicorn app.main:app --port 8000 --reload
```

**That's it!** Your local bot uses free Ollama LLM.

---

## **Part 2: Global Deployment to HuggingFace Spaces (FREE)**

### Step 1: Get Google API Key (2 minutes)

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Save it somewhere safe

### Step 2: Create HuggingFace Account (2 minutes)

1. Go to: https://huggingface.co
2. Sign up (free)
3. Go to: https://huggingface.co/spaces
4. Click "Create new Space"

### Step 3: Create Space Settings

- **Space Name**: `medical-rag-bot` (or whatever you like)
- **License**: OpenRAIL
- **Space SDK**: Docker
- **Visibility**: Public (so anyone can use it)

### Step 4: Push Code to HuggingFace

```bash
# Navigate to your project
cd c:\Users\swaro\Downloads\RAG-Based-Health-Bot-main

# Initialize git (if not already done)
git init

# Add HuggingFace remote
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot

# The URL format is: https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME

# Add all files
git add .

# Commit
git commit -m "Initial commit: Medical RAG Bot"

# Push to HuggingFace
git push -u origin main
```

### Step 5: Add Secrets in HuggingFace Dashboard

1. Go to your Space on HuggingFace
2. Click "Settings" (gear icon)
3. Scroll to "Repository secrets"
4. Add TWO secrets:

| Secret Name | Value |
|-------------|-------|
| `LLM_PROVIDER` | `google` |
| `GOOGLE_API_KEY` | (your API key from Step 1) |

5. Click "Save"

### Step 6: Wait for Deployment (2-5 minutes)

HuggingFace will automatically:
- Build the Docker image
- Install dependencies
- Start your bot
- Give you a public URL!

Once built, you'll see:
```
✅ Space is running at: https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot
```

---

## **Part 3: Switching Between Local & Global**

### 🏠 **Local Development (Ollama)**
```bash
# Use .env.local
cp .env.local .env
python -m uvicorn RAG-Based-Health-Bot-main.app.main:app --port 8000
# Visit: http://localhost:8000
```

### 🌍 **Global Deployment (Google)**
```bash
# Push to HuggingFace (already has Google API key in secrets)
git push origin main
# HuggingFace automatically rebuilds and deploys
# Visit: https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot
```

---

## **Troubleshooting**

### ❌ "Container build failed"
1. Check the build logs in HuggingFace Space settings
2. Most common: Missing dependencies in `requirements.txt`
3. Solution: Make sure all cloud LLM packages are installed:
   ```bash
   pip install langchain-google-genai langchain-openai langchain-anthropic
   pip freeze > requirements.txt
   git push
   ```

### ❌ "API key not working"
1. Make sure you added it to HuggingFace Secrets (not in `.env` file!)
2. Double-check the key is correct
3. Try generating a new key from: https://makersuite.google.com/app/apikey

### ❌ "App crashes on startup"
1. Check logs in Space settings → Logs
2. Usually means ChromaDB folder permissions issue
3. HuggingFace should handle this automatically

---

## **Files You Need**

✅ `Dockerfile` - For HuggingFace Spaces
✅ `.env.local` - Local development with Ollama
✅ `.env.google` - Reference for Google settings
✅ `requirements.txt` - All dependencies (including cloud LLMs)
✅ `DEPLOYMENT.md` - Additional deployment info

---

## **Cost Breakdown**

| Setup | Monthly Cost | LLM |
|-------|-------------|-----|
| **Local (Your Computer)** | $0 | Ollama |
| **HuggingFace Spaces** | $0 | Google Gemini (free tier) |
| **Total** | **$0** | ✅ |

---

## **Next Steps**

1. ✅ Get Google API key from: https://makersuite.google.com/app/apikey
2. ✅ Create HuggingFace account: https://huggingface.co
3. ✅ Create a Space: https://huggingface.co/spaces/new
4. ✅ Run the `git push` commands above
5. ✅ Add secrets in HuggingFace dashboard
6. ✅ Wait for deployment
7. ✅ Share your space URL with anyone! 🎉

---

## **Example URLs After Deployment**

- **Local**: `http://localhost:8000`
- **Global**: `https://huggingface.co/spaces/your-username/medical-rag-bot`

Anyone can now use your medical RAG bot from anywhere in the world! 🌍

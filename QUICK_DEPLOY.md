# 🎯 Quick Deploy Checklist

## **Before You Start** ✅
- [ ] Google API key ready? Get it: https://makersuite.google.com/app/apikey
- [ ] HuggingFace account created? https://huggingface.com
- [ ] Git installed on your computer?

---

## **5-Minute Deployment (Copy-Paste Commands)**

### **Step 1: Get Google API Key**
```
Visit: https://makersuite.google.com/app/apikey
Click: "Create API Key"
Copy the key (you'll need it in Step 4)
```

### **Step 2: Create HuggingFace Space**
```
1. Visit: https://huggingface.co/spaces/new
2. Space Name: medical-rag-bot
3. License: OpenRAIL
4. Space SDK: Docker
5. Visibility: Public
6. Click "Create Space"
```

You'll get a URL like: `https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot`

### **Step 3: Push Code to HuggingFace**

```bash
# Navigate to your project root
cd "c:\Users\swaro\Downloads\RAG-Based-Health-Bot-main"

# Initialize git (if needed)
git init

# Add HuggingFace as remote
# Replace YOUR-USERNAME with your actual username!
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot

# Stage all files
git add .

# Commit
git commit -m "Deploy medical RAG bot with Google Gemini"

# Push to HuggingFace
git push -u origin main

# When prompted for password, use your HuggingFace API token from:
# https://huggingface.co/settings/tokens
```

### **Step 4: Add Secrets in HuggingFace Dashboard**

Go to: `https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot/settings`

Under "Repository secrets", add:

```
Name: LLM_PROVIDER
Value: google

Name: GOOGLE_API_KEY  
Value: (paste your key from Step 1)
```

Click "Save"

### **Step 5: Wait for Build** ⏳

- Go to your Space page
- Wait 2-5 minutes for Docker build to complete
- Once done, you'll see a green "Running" status
- Click the app link to see your bot live! 🎉

---

## **Your Bot is Now LIVE!** 🚀

Share this URL with anyone:
```
https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot
```

**It's completely free and works with Google's free LLM tier!**

---

## **To Update Your Bot Later**

```bash
# Make changes to code

# Commit and push
git add .
git commit -m "Update: [your changes]"
git push origin main

# HuggingFace automatically rebuilds!
```

---

## **To Keep Using Ollama Locally**

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run your bot with local Ollama
cd RAG-Based-Health-Bot-main
python -m uvicorn app.main:app --port 8000

# Visit: http://localhost:8000
```

Both work independently - no conflicts! ✅

---

## **Troubleshooting**

**"git remote add origin failed"**
```bash
# You already added it, update instead:
git remote set-url origin https://huggingface.co/spaces/YOUR-USERNAME/medical-rag-bot
```

**"Repository not found"**
- Make sure you created the Space first (Step 2)
- Check your username is correct

**"Build failed"**
- Check the build logs in Space settings
- Usually a Python dependency issue
- Try: `pip install langchain-google-genai` and update requirements.txt

**"API key not working"**
- Add it to Space Secrets, NOT to .env file
- Get a fresh key: https://makersuite.google.com/app/apikey

---

## **Questions?**

- HuggingFace Spaces docs: https://huggingface.co/docs/hub/spaces
- Google Gemini API: https://ai.google.dev
- Your bot is ready to serve the world! 🌍

# SP-LineBot v1.0
**Production-ready Line Bot AI Assistant for Auto Repair Inventory**

Local-first architecture with Ollama (primary) + Gemini (fallback), multimodal support (image OCR, voice STT), Google Drive integration, and admin controls.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB RAM (4GB Ollama + 4GB app), GTX 1650 CUDA (optional)
- Windows/macOS/Linux
- Ollama (https://ollama.ai) — manual start only
- GitHub CLI optional (for gh repo create)

### Setup (5 minutes)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure .env:**
   ```
   LINE_CHANNEL_SECRET=your_line_secret
   LINE_CHANNEL_ACCESS_TOKEN=your_line_token
   GOOGLE_SERVICE_ACCOUNT_JSON=google-service-account.json
   GEMINI_API_KEY=your_gemini_key
   ADMIN_PIN_HASH=<sha256_hash_of_pin>
   OLLAMA_HOST=http://127.0.0.1:11434
   PORT=8000
   ```

3. **Start Ollama (separate terminal):**
   ```bash
   ollama serve
   ```

4. **Start app:**
   ```bash
   python main.py
   ```
   - Server runs on `0.0.0.0:8000`
   - GET `/health` for status check

5. **Expose via ngrok:**
   ```bash
   ngrok http 8000
   ```
   Configure Line webhook to ngrok URL `/callback`

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | FastAPI webhook handler (Line, text/image/voice) |
| `local_llm.py` | Ollama client + intent parsing + spam detection |
| `multimodal.py` | Image OCR (Thai/Eng), voice STT (Vosk), embedding |
| `drive_handler.py` | Google Drive scan, batch embedding with Chroma |
| `admin_commands.py` | User management (add/delete/list), PIN auth |
| `analyze_logs.py` | Log analysis script with Gemini insights |
| `.gitignore` | Safety: excludes secrets, caches, large files |
| `requirements.txt` | All dependencies (FastAPI, torch, Ollama, etc.) |

---

## 🔐 Security

✅ **No Hardcoded Secrets**
- All keys in `.env` (excluded from git via .gitignore)
- Pin-based admin auth (SHA256 hashing)

✅ **File Safety**
- Ignores: `.env`, `google-service-account*.json`, `*.db`, `logs/`, `vector_db/`
- Temp files cleaned automatically

✅ **API Rate Limiting**
- Gemini: Free tier (15 RPM, 1500 RPD)
- Message spam detection included
- Concurrent image processing capped at 5

---

## 📊 Scalability Features

✅ **40 Users Support**
- Per-user Chroma collections
- Message history decay (60s)
- Batch intent parsing
- Threaded multimodal processing

✅ **Low-Resource Design**
- CUDA optional (CPU fallback)
- SentenceTransformer caching
- Ollama local (no network overhead)
- Processed files cache (no re-embedding)

---

## 🤖 AI Architecture

### Intent Chain
```
User Message → Ollama (local LLM) → Intent + Confidence
             ↓ (if conf < 0.7)
           Gemini API → Better Response
```

### Multimodal Pipeline
```
Image → EasyOCR (Thai/Eng) → Text Extraction → Embed
Voice → Vosk STT → Text → Intent → Response
```

### Drive Integration
```
Google Drive → Scan (recursive) → Extract Text → Chunk & Embed → Chroma Store
```

---

## 🛠️ Admin Commands

(Requires PIN auth in context)

- **Add User:** `ADMIN_ADD_USER [user_id] [name]`
  - Auto-creates Drive folder
- **Delete User:** `ADMIN_DEL_USER [user_id]`
- **List Users:** `ADMIN_LIST_USERS`

---

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```
Response: GPU status, Ollama health

### Logs
- Application: `logs/sp_linebot.log`
- Analysis script: `python analyze_logs.py`

### User Database
- Location: `drive_sync/users.json`
- Processed files: `drive_sync/processed_files.json`

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama connection timeout | Ensure `ollama serve` is running in separate terminal |
| "No Line credentials" | Check LINE_CHANNEL_SECRET/TOKEN in .env |
| GPU not detected | CPU fallback active, performance will be slower |
| Gemini quota exceeded | Wait or upgrade to paid tier |
| Chroma connection error | Ensure chromadb installed (`pip install chromadb`) |

---

## 📝 Example Usage (Line Bot)

**User:** "Stock khmer parts?"  
**Bot:** Runs intent parser → "INVENTORY_LOOKUP" → Queries Drive → Returns availability

**User:** [Sends image of repair order]  
**Bot:** OCR extract → Embed in Chroma → Suggest matching tasks

**User:** [Sends voice: "Schedule service"]  
**Bot:** Vosk STT → Intent → Calendar integration

---

## 🔧 Development

### Add New Intent
1. Update `FIXED_COMMANDS` in `local_llm.py`
2. Add handler in `main.py` `handle_intent()`
3. Restart app

### Extend Multimodal
- Add language: `languages=['th', 'en', 'zh']` in `multimodal.py`
- Add Vosk model: Download from alphacephei.com

### Custom Embedding Model
- Replace `all-MiniLM-L6-v2` in `drive_handler.py` & `local_llm.py`
- Options: `sentence-bert`, `multilingual-MiniLM`, etc.

---

## 📦 Dependencies

See `requirements.txt` for full list:
- **FastAPI/Uvicorn** — Web framework
- **Line Bot SDK** — Line messaging
- **Torch** — Deep learning (GPU support)
- **SentenceTransformers** — Embedding model
- **Chroma** — Vector database
- **EasyOCR** — Image text recognition
- **Vosk** — Offline STT
- **Ollama** — Local LLM client
- **Google APIs** — Drive, Generative AI

---

## 📄 License

Proprietary — SP-LineBot (Internal use)

---

## 🎯 Roadmap

- [ ] Thai language tokenization improvements
- [ ] Multi-turn conversation memory
- [ ] Webhook signature caching
- [ ] Mobile app integration
- [ ] Dashboard for log analysis
- [ ] ABM (Account-Based Messaging)

---

## 📞 Support

Issues? Check logs:
```bash
tail -f logs/sp_linebot.log
```

Last updated: **March 13, 2026** | UTC+7 (Bangkok)

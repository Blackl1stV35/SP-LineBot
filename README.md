# 🏭 SP-LineBot: Factory Analytics & Agentic Inventory Bot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Storage-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-1.1.0-purple.svg)

A production-ready, highly modular LINE Bot designed for factory environments.

</div>

---

## 📋 Overview

**SP-LineBot** features an **Agentic Architecture** that integrates Google Drive inventory tracking, real-time database write-backs, conversational RAG, and an intelligent local LLM fallback system — all accessible via LINE chat.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 🤖 **Agentic Intent Routing** | Automatically decides whether to search inventory, write new memories, or chat normally based on user input |
| 📊 **Semantic Data Unrolling** | Transforms complex 2D Excel/CSV inventory matrices into highly searchable text chunks |
| 💾 **Real-Time Memory Write-Back** | Users can tell the bot to remember new inventory draws via LINE chat, instantly updating the vector database |
| 🔁 **Resilient Hybrid LLM** | Uses Google Gemini API as the primary engine with seamless fallback to a local Ollama model (Typhoon) on rate limits |
| 🎤 **Multimodal Input** | Supports both text and Thai voice commands via Vosk STT |
| 📈 **Live Web Dashboard** | A Streamlit UI that reads the vector memory in real-time to visualize inventory requisitions and employee statistics |

---

## ⚡ v1.1.0 — Hardware & Performance Optimizations

- **Singleton Database Connection** — ChromaDB is now initialized once at startup, preventing I/O bottlenecks and reducing RAM consumption.
- **Asynchronous Routing** — Replaced blocking API calls with `async` operations, allowing the FastAPI server to handle multiple user requests simultaneously while waiting for the LLM.
- **Model VRAM Caching** — `SentenceTransformer` embedding models are loaded into VRAM once, drastically speeding up data parsing and semantic search.
- **Strict Prompt Formatting** — Local Typhoon/Ollama outputs are now wrapped in strict instruction templates to prevent prompt-leaking into user chat.

---

## 📂 Project Structure

```text
SP-LineBot/
├── src/                        # Main Application Code
│   ├── api/                    # FastAPI endpoints and LINE webhook
│   ├── agent/                  # AI Router & LLM Generation Logic
│   ├── services/               # Google Drive, File Scanning, & Audio processing
│   └── db/                     # Vector Database & Singleton Client
├── ui/
│   └── dashboard.py            # Streamlit web interface
├── tests/                      # Testing scripts
├── .env                        # Environment variables
└── requirements.txt            # Project dependencies
```

---

## 🚀 Setup & Installation

### 1. Prerequisites

- Python **3.10+**
- [Ollama](https://ollama.com/) installed and running locally with the Typhoon model
- Google Cloud Service Account JSON (for Drive access)
- LINE Developers Messaging API credentials

### 2. Environment Variables

Create a `.env` file in the root directory:

```ini
LINE_CHANNEL_SECRET=your_secret
LINE_CHANNEL_ACCESS_TOKEN=your_token
GOOGLE_API_KEY=your_gemini_key
SERVICE_ACCOUNT_FILE=path/to/your/google_credentials.json
DRIVE_FOLDER_ID=your_target_google_drive_folder_id
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Bot (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

> Use `ngrok http 8000` to expose your local server to the LINE Developer Console.

### 5. Run the Analytics Dashboard

```bash
streamlit run ui/dashboard.py
```

### 6. Start Ollama (if not running)

```bash
ollama serve
ollama run typhoon
```

---

## 🛠️ Usage Flow

```
1. Ingest   →   Upload an Excel inventory sheet to the connected Google Drive folder.

2. Scan     →   Type  สแกนไดรฟ์  in LINE chat.
                The bot downloads, parses, and memorizes the data.

3. Query    →   Ask questions in natural Thai:
                "เดือน 2 ใครเบิกสีพ่นอุดดำเงาบ้าง"

4. Update   →   Add data dynamically:
                "จดบันทึกย้อนหลัง สมชายเบิกทินเนอร์ 5 แกลลอน"

5. Analyze  →   Open the Streamlit Dashboard to see newly logged data appear instantly.
```

---

## 🔧 Troubleshooting

<details>
<summary><strong>Ollama not responding?</strong></summary>

```bash
ollama serve
ollama run typhoon
```

</details>

<details>
<summary><strong>Embeddings running slow?</strong></summary>

Check GPU/CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

</details>

<details>
<summary><strong>Database corrupted or stale?</strong></summary>

Delete the local vector cache and rescan:

```bash
rm -rf chroma_data/
```

Then re-run the scan command in LINE chat.

</details>

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

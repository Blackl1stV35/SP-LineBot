# 🏭 Factory Analytics & Inventory LINE Bot (v1.0)

A production-ready, multimodal LINE Bot designed for factory environments. It features an Agentic Architecture that integrates Google Drive inventory tracking, real-time database write-backs, conversational RAG, and an intelligent local LLM fallback system.

## 🌟 Key Features

* **Agentic Intent Routing:** Automatically decides whether to search inventory, write new memories, or chat normally based on the user's message.
* **Semantic Data Unrolling:** Transforms complex, dense 2D Excel/CSV inventory matrices into highly searchable semantic text chunks.
* **Real-Time Memory Write-Back:** Users can tell the bot to remember new inventory draws via LINE chat, and it instantly updates the ChromaDB vector database.
* **Resilient Hybrid LLM:** Uses Google Gemini API as the primary engine. If the API hits a rate limit (429), it seamlessly falls back to a local Ollama model (Typhoon) with zero downtime.
* **Multimodal Input:** Supports both text and Thai voice commands (via Vosk STT).
* **Live Web Dashboard:** A Streamlit web UI that reads the vector memory in real-time to visualize inventory requisitions and employee statistics.

## 📂 Project Structure

\`\`\`text
├── main.py              # FastAPI server & LINE Webhook handler
├── agentic_router.py    # Analyzes user intent and routes to tools
├── drive_handler.py     # Connects to Google Drive, downloads files
├── drive_scanner.py     # Parses Excel/CSV into semantic RAG chunks
├── db_updater.py        # Handles manual memory write-backs to ChromaDB
├── local_llm.py         # Manages Gemini API and Local Ollama fallback
├── multimodal.py        # Handles Vosk Speech-to-Text for voice messages
├── dashboard.py         # Streamlit Web UI for inventory analytics
├── inspect_db.py        # CLI utility to verify ChromaDB chunk formatting
├── admin_commands.py    # System commands (e.g., "สแกนไดรฟ์")
└── .env                 # Environment variables (Keys, Tokens)
\`\`\`

## 🚀 Setup & Installation

**1. Prerequisites:**
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running locally with the Typhoon model (`ollama run typhoon`).
* Google Cloud Service Account JSON (for Drive access).
* LINE Developers Messaging API keys.

**2. Environment Variables (`.env`):**
\`\`\`ini
LINE_CHANNEL_SECRET=your_secret
LINE_CHANNEL_ACCESS_TOKEN=your_token
GOOGLE_API_KEY=your_gemini_key
SERVICE_ACCOUNT_FILE=path/to/your/google_credentials.json
DRIVE_FOLDER_ID=your_target_google_drive_folder_id
\`\`\`

**3. Install Dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**4. Run the Bot:**
\`\`\`bash
uvicorn main:app --host 0.0.0.0 --port 8000
# Expose via Ngrok if testing locally: ngrok http 8000
\`\`\`

**5. Run the Dashboard:**
\`\`\`bash
streamlit run dashboard.py
\`\`\`

## 🛠️ Usage Flow
1. Upload an Excel inventory sheet to the connected Google Drive.
2. Type `สแกนไดรฟ์` in the LINE chat. The system will download, parse, and memorize the data.
3. Ask questions in natural Thai: *"เดือน 2 ใครเบิกสีพ่นอุดดำเงาบ้าง"*
4. Add data dynamically: *"จดบันทึกย้อนหลัง สมชายเบิกทินเนอร์ 5 แกลลอน"*
5. Check the Dashboard to see the newly logged data appear instantly.

## 🔜 Roadmap (v1.1)
* Fix local LLM (Typhoon) prompt leaking/formatting in chat responses.
* Increase Agentic Router timeout thresholds for local hardware.
* Deploy to a dedicated cloud server.

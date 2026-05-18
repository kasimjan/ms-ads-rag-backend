# UChicago MS ADS RAG Backend

This backend was converted from the `genai_rag.ipynb` notebook.

## Folder structure

```text
.
├── main.py
├── rag_engine.py
├── requirements.txt
├── render.yaml
├── .env.example
└── data/
    └── txt/
        ├── your_ms_ads_page_1.txt
        ├── your_ms_ads_page_2.txt
        └── ...
```

## Do I need ChromaDB from Google Drive?

Recommended: copy the `.txt` files from your Google Drive folder into `data/txt/`.

You do **not** need to copy the existing ChromaDB folder if you have all `.txt` source files. The backend will rebuild Chroma automatically on first run if `chromadb_genai_midterm` is missing or empty.

Copying ChromaDB is optional. It can make startup faster, but it is more fragile because Chroma versions and file paths can differ between Colab, local machine, and Render.

## Local test

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your real `OPENAI_API_KEY`.

Mac/Linux:

```bash
export OPENAI_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Run:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
http://localhost:8000/docs
```

Test:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the core courses in the MS in Applied Data Science program?\"}"
```

## Render deployment

1. Push this folder to GitHub.
2. In Render, create `New Web Service`.
3. Connect your GitHub repo.
4. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable:
   - `OPENAI_API_KEY = your real key`
6. Deploy.

## Lovable

Use your Render endpoint:

```text
POST https://your-service-name.onrender.com/chat
```

Request:

```json
{
  "question": "What is the capstone project?"
}
```

Response:

```json
{
  "question": "...",
  "answer": "...",
  "sources": [...]
}
```

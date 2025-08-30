# 📰 News Research Tool

This is a **Streamlit app** that lets you load news articles from URLs, build embeddings with **Google Generative AI (Gemini)**, store them in a **FAISS vector database**, and then ask questions with **retrieval-augmented generation (RAG)**.

---

## 🚀 Features
- Load up to 3 news article URLs at a time
- Split text into manageable chunks
- Generate embeddings using **Google Generative AI Embeddings**
- Store vectors locally with **FAISS**
- Ask natural language questions about the articles
- Get answers with **citations** to the relevant sources

---

## 📦 Installation

Clone this repo and set up a virtual environment:

```bash
git clone <your-repo-url>
cd Equity-Research-Analysis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

You need a **Google Gemini API key**.  
Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the provided URL in your browser (default: `http://localhost:8501`).

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) → UI
- [LangChain](https://www.langchain.com/) → LLM orchestration
- [Google Generative AI](https://ai.google.dev/) → Embeddings + Chat LLM
- [FAISS](https://faiss.ai/) → Vector database

---

## 📂 Project Structure

```
Equity-Research-Analysis/
│── app.py              # Main Streamlit app
│── test.py             # Simple FAISS/embeddings test script
│── requirements.txt    # Dependencies
│── .env                # API key (not committed to git)
│── faiss_index/        # Saved FAISS vectorstore
```

---

## 📝 Usage Flow

1. Enter up to **3 URLs** in the sidebar.
2. Click **Process URLs**:
   - Articles are loaded and split
   - Embeddings are generated with Gemini
   - A FAISS index is created and saved
3. Ask a question in the input box.
4. Get an answer with sources.

---

## ⚡ Troubleshooting

- **`RuntimeError: There is no current event loop`**  
  → Fixed by ensuring an asyncio loop exists (already patched in `app.py`).

- **`TypeError: VectorStore.as_retriever() takes 1 positional argument but 2 were given`**  
  → Call `.as_retriever()` with no embeddings (already patched in `app.py`).

- **FAISS index not found**  
  → Make sure you’ve processed URLs at least once before asking questions.

---
# Equity-Research-Analysis-

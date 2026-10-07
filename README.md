# RAG — Upload a PDF and Ask Questions

This version intentionally stays simple.

## Files

- `simple_rag.py` — rag
- `rag_engine.py` — the same RAG steps extracted into small functions so the Streamlit UI stays clean.
- `app.py` — the Streamlit interface.
- `data/` — uploaded PDFs are stored here.

## Run

```powershell
streamlit run app.py
```

Then:

1. Upload one or more PDFs.
2. Click **Build knowledge base**.
3. Ask questions in the chat box.
4. Open **Sources & evidence** to see what the retriever found.

## Why it should be faster

The expensive Hugging Face embedding model is cached with `st.cache_resource`.

The vector database is built **once after you upload the document**. It is not rebuilt every time you ask a question.

The default Groq model is `openai/gpt-oss-20b` to keep the app responsive.

The RAG pipeline is still exactly the simple lecture pipeline:

PDF → Load → Chunk → Embed → Chroma → Retrieve → Context → LLM → Answer

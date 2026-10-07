import time
from pathlib import Path

import streamlit as st

from rag_engine import (
    build_vector_store,
    create_embedding_model,
    create_llm,
    generate_answer,
    load_documents,
    retrieve,
    save_uploaded_files,
)

# --------------------------------------------------
# Page
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Simple polished UI
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(61,214,178,.10), transparent 30%),
        radial-gradient(circle at 5% 20%, rgba(79,145,255,.08), transparent 28%),
        #07111c;
    color: #eef5ff;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 6rem;
}

[data-testid="stSidebar"] {
    background: #091522;
    border-right: 1px solid rgba(255,255,255,.08);
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:22px;
}

.logo {
    width:44px;
    height:44px;
    border-radius:14px;
    display:grid;
    place-items:center;
    background:linear-gradient(135deg,#3dd6b2,#5b9cff);
    color:#06131f;
    font-weight:900;
    font-size:20px;
}

.title {
    font-size:18px;
    font-weight:800;
}

.muted {
    color:#8498ad;
    font-size:12px;
}

.hero {
    padding:34px 30px;
    border:1px solid rgba(255,255,255,.08);
    border-radius:22px;
    background:linear-gradient(135deg,rgba(16,34,53,.94),rgba(9,23,38,.94));
    margin:12px 0 22px;
}

.badge {
    display:inline-block;
    padding:5px 9px;
    border-radius:999px;
    background:rgba(61,214,178,.09);
    border:1px solid rgba(61,214,178,.25);
    color:#66e1c4;
    font-size:10px;
    font-weight:800;
    letter-spacing:.08em;
}

h1 {
    font-size:36px !important;
    margin:10px 0 8px !important;
}

[data-testid="stChatMessage"] {
    background:rgba(13,28,45,.72);
    border:1px solid rgba(255,255,255,.07);
    border-radius:17px;
    margin:8px 0;
}

[data-testid="stChatInput"] > div {
    background:#0b1928 !important;
    border:1px solid #28415a !important;
    border-radius:16px !important;
}

.stButton > button {
    border-radius:12px;
    border:1px solid rgba(255,255,255,.09);
    background:#102338;
    color:#dce8f4;
}

.stButton > button:hover {
    border-color:#3dd6b2;
    color:white;
}

.source {
    padding:12px;
    margin:7px 0;
    border-radius:12px;
    border:1px solid rgba(255,255,255,.07);
    background:#091522;
}

.footer {
    text-align:center;
    color:#536a80;
    font-size:11px;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# --------------------------------------------------
# Cached expensive resources
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return create_embedding_model()

@st.cache_resource(show_spinner=False)
def get_llm():
    return create_llm()

# --------------------------------------------------
# Sidebar: upload + settings
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="logo">◈</div>
        <div>
            <div class="title">DocuMind RAG</div>
            <div class="muted">Ask questions about your document</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Upload document")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        names = ", ".join(f.name for f in uploaded_files)
        st.caption(names)

        if st.button("Build knowledge base", use_container_width=True):
            with st.spinner("Reading and indexing document..."):
                try:
                    save_uploaded_files(uploaded_files)
                    documents = load_documents(uploaded_files)
                    embedding_model = get_embedding_model()

                    vector_store, chunk_count = build_vector_store(
                        documents,
                        embedding_model,
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.chunk_count = chunk_count
                    st.session_state.document_name = names
                    st.session_state.messages = []

                    st.success(f"Ready • {chunk_count} chunks indexed")
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("### 2. Retrieval")

    top_k = st.slider(
        "Number of evidence chunks",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.session_state.vector_store:
        st.markdown(
            f"🟢 **Ready**  \n"
            f"`{st.session_state.chunk_count}` chunks indexed"
        )
    else:
        st.markdown("⚪ **No document loaded**")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Simple RAG: Load → Chunk → Embed → Retrieve → Generate"
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <span class="badge">RETRIEVAL-AUGMENTED GENERATION</span>
    <h1>Ask your document anything.</h1>
    <div class="muted" style="font-size:14px;max-width:720px;line-height:1.7;">
        Upload a PDF, build its knowledge base once, then ask questions.
        Answers are generated from retrieved passages in your document.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Chat history
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("documents"):
            with st.expander("Sources & evidence"):
                for i, doc in enumerate(message["documents"], 1):
                    source = Path(
                        doc.metadata.get("source", "Unknown")
                    ).name
                    page = doc.metadata.get(
                        "page_label",
                        doc.metadata.get("page", "?"),
                    )

                    st.markdown(
                        f'<div class="source"><b>📄 {source}</b>'
                        f'<div class="muted">Page {page}</div>'
                        f'<div style="margin-top:8px;color:#b8c7d7;">'
                        f'{doc.page_content}</div></div>',
                        unsafe_allow_html=True,
                    )

# --------------------------------------------------
# Welcome state
# --------------------------------------------------

if not st.session_state.messages:
    if not st.session_state.vector_store:
        st.info("Upload a PDF from the sidebar to start.")
    else:
        st.markdown(
            f"**Loaded:** `{st.session_state.document_name}`"
        )

        a, b, c = st.columns(3)

        prompts = [
            "Summarize this document.",
            "What are the main rules discussed?",
            "What are the most important restrictions?",
        ]

        for col, prompt in zip((a, b, c), prompts):
            if col.button(prompt, use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()

# --------------------------------------------------
# Question
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded document..."
)

if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    if not st.session_state.vector_store:
        st.warning("Please upload and index a document first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            start = time.perf_counter()

            # Retrieval only. The embedding model and vector store already exist.
            documents = retrieve(
                st.session_state.vector_store,
                question,
                top_k,
            )

            retrieval_time = time.perf_counter() - start

            llm = get_llm()

            generation_start = time.perf_counter()

            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    llm,
                    question,
                    documents,
                )

            generation_time = time.perf_counter() - generation_start
            total_time = time.perf_counter() - start

            st.markdown(answer)

            st.caption(
                f"Retrieval {retrieval_time:.2f}s · "
                f"Generation {generation_time:.2f}s · "
                f"Total {total_time:.2f}s"
            )

            with st.expander("Sources & evidence"):
                for i, doc in enumerate(documents, 1):
                    source = Path(
                        doc.metadata.get("source", "Unknown")
                    ).name
                    page = doc.metadata.get(
                        "page_label",
                        doc.metadata.get("page", "?"),
                    )

                    st.markdown(
                        f"**{i}. {source} — page {page}**"
                    )
                    st.caption(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "documents": documents,
            })

        except Exception as exc:
            st.error(str(exc))

st.markdown(
    '<div class="footer">Simple RAG · Your document stays the knowledge source</div>',
    unsafe_allow_html=True,
)

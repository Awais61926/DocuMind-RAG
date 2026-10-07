import hashlib
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = """You are a helpful assistant for the uploaded document.

Answer the question using ONLY the retrieved context.
Do not invent or assume information.

If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def create_llm():
    groq_key = os.getenv("GROQ_KEY")

    if not groq_key:
        raise ValueError("GROQ_KEY was not found in .env.")

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_key,
        temperature=0,
        max_tokens=350,
    )


def save_uploaded_files(uploaded_files):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Keep the uploaded document available in the project.
    for old_file in data_dir.glob("*.pdf"):
        old_file.unlink()

    for uploaded_file in uploaded_files:
        (data_dir / uploaded_file.name).write_bytes(uploaded_file.getvalue())


def load_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        temp_path = Path("data") / uploaded_file.name
        loader = PyPDFLoader(str(temp_path))
        documents.extend(loader.load())

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def build_vector_store(documents, embedding_model):
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDF.")

    # In-memory Chroma: the index is built once for the uploaded document.
    collection_name = "uploaded_document_" + hashlib.sha1(
        str(len(chunks)).encode()
    ).hexdigest()[:12]

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
    ), len(chunks)


def retrieve(vector_store, question, top_k=3):
    return vector_store.similarity_search(question, k=top_k)


def generate_answer(llm, question, documents):
    if not documents:
        return "I could not find this information in the uploaded document."

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    response = llm.invoke(prompt)
    return response.content

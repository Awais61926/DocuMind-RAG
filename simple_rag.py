## importing libraries
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# --------------------------------------------------
# loading env
# --------------------------------------------------

load_dotenv()

groq_key = os.getenv("GROQ_KEY")

print("Groq Key Loaded:", groq_key is not None)


# --------------------------------------------------
# loading ALL PDFs from data folder
# --------------------------------------------------

documents = []

pdf_files = list(Path("data").glob("*.pdf"))

for pdf_file in pdf_files:
    print("Loading:", pdf_file)

    loader = PyPDFLoader(str(pdf_file))
    pdf_documents = loader.load()

    documents.extend(pdf_documents)


print("Total PDFs:", len(pdf_files))
print("Total document pages:", len(documents))


# --------------------------------------------------
# chunking the data
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

if chunks:
    print("Chunk Content:", chunks[0].page_content)
    print("Chunk Meta Data:", chunks[0].metadata)


# --------------------------------------------------
# embedding
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# create Chroma vector store
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("Vector loaded successfully!")


# --------------------------------------------------
# test question
# --------------------------------------------------

question = "What restrictions apply to WAPDA employees regarding gifts?"


# --------------------------------------------------
# similarity search
# --------------------------------------------------

results = vector_store.similarity_search(
    question,
    k=3
)


for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)


# --------------------------------------------------
# retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

retrieved_docs = retriever.invoke(question)


print("\n--- Retriever Results ---")

for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Retrieved Document {i + 1} ---")
    print(doc.page_content)
    print(doc.metadata)


# --------------------------------------------------
# build context
# --------------------------------------------------

context = ""

for doc in retrieved_docs:
    context = context + doc.page_content + "\n\n"


# --------------------------------------------------
# prompt
# --------------------------------------------------

prompt = f"""
You are a helpful assistant for WAPDA employee rules.

Answer the question using only the context provided below.

If the answer is not available in the context, say:
"I could not find this information in the provided WAPDA rules."

Context:
{context}

Question:
{question}

Answer:
"""


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_key,
    temperature=0,
    max_tokens=350
)


# --------------------------------------------------
# generate answer
# --------------------------------------------------

response = llm.invoke(prompt)

answer = response.content


# --------------------------------------------------
# display answer
# --------------------------------------------------

print("\nAnswer:")
print(answer)


# --------------------------------------------------
# display sources
# --------------------------------------------------

print("\nSources:")

for doc in retrieved_docs:
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page_label", doc.metadata.get("page", "Unknown"))

    print(f"- {source} | Page {page}")

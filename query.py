# query.py – Phase 3.2 upgrade: metadata filtering

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

load_dotenv()

# === Config ===
CHROMA_DIR = "vectorstore/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"  # or gpt-4 if enabled

# === Setup ===
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# === User input ===
query = input("💬 Ask a question: ").strip()
ticker = input("🏷️ Filter by ticker (e.g. TSLA, AAPL), or leave blank: ").strip().upper()
year = input("📅 Filter by year (e.g. 2024), or leave blank: ").strip()
quarter = input("🗓️ Filter by quarter (e.g. Q3), or leave blank: ").strip().upper()

# === Metadata filter ===
metadata_filter = {"$and": []}
if ticker: metadata_filter["$and"].append({"ticker": ticker})
if year: metadata_filter["$and"].append({"year": year})
if quarter: metadata_filter["$and"].append({"quarter": quarter})
if not metadata_filter["$and"]:
    metadata_filter = {}  # No filters provided

# === Retriever setup ===
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": metadata_filter}
)

# Optional: compressor to reduce context bloat (keeps best sentences)
# llm = ChatOpenAI(model_name=LLM_MODEL)
# compressor = LLMChainExtractor.from_llm(llm)
# retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# === Run query ===
result = qa_chain.invoke({"query": query})

# === Output ===
print("\n📊 ANSWER:\n", result["result"])
print("\n📚 SOURCES:")

for i, doc in enumerate(result["source_documents"]):
    meta = doc.metadata
    print(f"- {meta.get('ticker', '?')} {meta.get('quarter', '')} {meta.get('year', '')} | Page: {meta.get('page', '?')} | File: {meta.get('file_path', '')}")

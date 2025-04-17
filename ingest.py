# ingest.py — Phase 3 upgrade: multi-file with metadata

import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# CONFIG
DATA_DIR = "data"
CHROMA_DIR = "vectorstore/chroma_db"

# Helper to extract metadata from filename
def extract_metadata(filename):
    base = Path(filename).stem.lower().replace("_", "-")
    parts = base.split("-")
    ticker, quarter, year = None, None, None

    for part in parts:
        if part.upper() in {"Q1", "Q2", "Q3", "Q4"}:
            quarter = part.upper()
        elif part.isdigit() and len(part) == 4:
            year = part
        elif len(part) <= 5 and part.isalpha():
            ticker = part.upper()

    if not (ticker and quarter and year):
        return None

    return {"ticker": ticker, "quarter": quarter, "year": year}



# Main ingestion logic
def load_and_chunk_with_metadata(file_path):
    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()
    metadata = extract_metadata(file_path)

    # Attach metadata to each document chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata.update(metadata)
    return chunks

def embed_all_chunks(all_chunks):
    print("🧠 Embedding documents...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(all_chunks, embedding=embedding, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    print(f"✅ Vectorstore saved to: {CHROMA_DIR}")

def main():
    all_chunks = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            print(f"📄 Processing: {file}")
            try:
                chunks = load_and_chunk_with_metadata(path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Skipped {file}: {e}")
    if all_chunks:
        embed_all_chunks(all_chunks)
    else:
        print("❌ No valid PDFs found to process.")

if __name__ == "__main__":
    main()

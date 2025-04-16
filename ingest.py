from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load OpenAI API key from .env
load_dotenv()

# 🔧 CONFIG
PDF_PATH = "data/TSLA-Q4-2024-Update.pdf"
CHROMA_DIR = "vectorstore/chroma_db"

def load_pdf(pdf_path):
    print(f"📄 Loading report: {pdf_path}")
    loader = UnstructuredPDFLoader(pdf_path)
    return loader.load()

def chunk_documents(docs):
    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def embed_chunks(chunks, persist_dir):
    print("🧠 Embedding & storing into ChromaDB...")
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"✅ Vectorstore saved at: {persist_dir}")

def main():
    docs = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    embed_chunks(chunks, CHROMA_DIR)

if __name__ == "__main__":
    main()

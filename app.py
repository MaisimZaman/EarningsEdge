import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# === Config ===
load_dotenv()
CHROMA_DIR = "vectorstore/chroma_db"
DATA_DIR = "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"

# === Ensure data directory exists ===
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# === Init Embedding + Vectorstore ===
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# === Streamlit UI ===
st.set_page_config(page_title="EarningsEdge", layout="wide")
st.title("📊 EarningsEdge – Financial Analyst AI")

# === File Upload ===
st.sidebar.header("📤 Upload Earnings PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF reports", type=["pdf"], accept_multiple_files=True)

def extract_metadata(filename):
    parts = Path(filename).stem.split("-")
    if len(parts) < 3:
        return None
    return {"ticker": parts[0], "quarter": parts[1], "year": parts[2]}

def save_uploaded_files(files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in files:
        file_path = Path(DATA_DIR) / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())

        metadata = extract_metadata(file.name)
        if not metadata:
            st.warning(f"❌ Skipped {file.name}: Invalid format.")
            continue

        loader = PyMuPDFLoader(str(file_path))
        raw_docs = loader.load()
        chunks = splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata.update(metadata)
        all_chunks.extend(chunks)
    return all_chunks

# === Ingest PDFs ===
if uploaded_files:
    with st.spinner("🔄 Ingesting and embedding..."):
        chunks = save_uploaded_files(uploaded_files)
        if chunks:
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            st.success("✅ Uploaded and embedded successfully.")

# === Sidebar Metadata Filters ===
st.sidebar.header("📂 Filter Documents")
ticker = st.sidebar.text_input("Ticker (e.g. TSLA)").strip().upper()
year = st.sidebar.text_input("Year (e.g. 2024)").strip()
quarter = st.sidebar.text_input("Quarter (e.g. Q3)").strip().upper()

metadata_filter = {}
filters = []
if ticker:
    filters.append({"ticker": ticker})
if year:
    filters.append({"year": year})
if quarter:
    filters.append({"quarter": quarter})
if filters:
    metadata_filter = {"$and": filters}

# === Chat Interface ===
st.markdown("## 💬 Chat with the Earnings Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    st.chat_message(role).markdown(content)

# User input
user_input = st.chat_input("Ask your question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    search_args = {"k": 5}
    if metadata_filter:
        search_args["filter"] = metadata_filter

    retriever = vectorstore.as_retriever(search_kwargs=search_args)
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)

    context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    prompt = f"""You are a financial analyst AI. Answer based on the following conversation and documents.
Conversation:
{context}

Answer the user's last question truthfully and concisely, using financial facts from the retrieved documents."""

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    with st.spinner("🤔 Thinking..."):
        result = qa_chain.invoke({"query": prompt})

    response = result["result"]
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.expander("📚 Sources used"):
        for doc in result["source_documents"]:
            meta = doc.metadata
            st.markdown(f"- **{meta.get('ticker', '?')} {meta.get('quarter', '')} {meta.get('year', '')}** – Page {meta.get('page', '?')} – *{os.path.basename(meta.get('file_path', ''))}*")

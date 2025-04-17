# 📊 EarningsEdge: Your AI-Powered Financial Analyst

EarningsEdge is an intelligent RAG-based (Retrieval-Augmented Generation) financial analyst powered by OpenAI and LangChain.  
It ingests real earnings reports (10-Qs, investor decks, earnings calls) and answers financial questions with source-grounded precision — like an analyst that never sleeps.

---

## 💼 Why I Built This

As part of my AI/data science career journey, I wanted to build something **more than a toy project**.  
EarningsEdge solves a real institutional problem: **automating financial intelligence** for analysts, hedge funds, and data-driven investors.

---

## 🧠 Core Features

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 📂 Upload Earnings Reports       | Upload any PDF 10-Q, 10-K, or earnings deck (e.g. `TSLA-Q4-2024.pdf`)       |
| 🧠 RAG Architecture              | GPT-3.5/4 answers only using real embedded documents                        |
| 🗂️ Metadata Filtering            | Ask questions scoped to ticker, quarter, or year                            |
| 💬 Chat Interface                | Persistent memory across multi-turn conversations                          |
| 📚 Source Traceability           | Every answer is backed by document citations (page, ticker, file)          |
| ⚡ Fast Local Embeddings         | Uses HuggingFace MiniLM (no external vector DB needed)                     |
| 🧪 PDF-Agnostic & Company-Agnostic | Works across different formatting styles (Tesla, Ford, Apple, etc.)         |

---

## 🖼️ Demo Preview


---

## 🚀 Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/earningsedge.git
cd earningsedge

from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"]  # This will be picked up by OpenAI client

# Load persisted vectorstore
vectorstore = Chroma(
    persist_directory="vectorstore/chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # “map_reduce” or “refine” for large contexts
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "What was Tesla’s total revenue in Q1 2024?"
result = qa_chain(query)

# Show result
print("\n📊 ANSWER:\n", result["result"])
print("\n📚 SOURCES:")
for doc in result["source_documents"]:
    print("-", doc.metadata)

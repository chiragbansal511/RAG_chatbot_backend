import os
import json
from flask import Flask, request, jsonify
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.chains import RetrievalQA

# Flask app
app = Flask(__name__)

# Paths
RESUME_DIR = "resumes"
MODEL_PATH = "model/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Auto-download model if not present
model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Mistral model...")
    os.makedirs("model", exist_ok=True)
    import subprocess
    subprocess.run(["wget", "-O", MODEL_PATH, model_url])

# Load documents
def load_documents(data_dir=RESUME_DIR):
    paths = Path(data_dir).glob("*.txt")
    docs = []
    for path in paths:
        loader = TextLoader(str(path))
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Create QA pipeline
def create_qa_pipeline():
    docs = load_documents()
    embeddings = GPT4AllEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.6,
        max_tokens=512,
        top_p=0.95,
        n_ctx=2048,
        verbose=False,
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa_chain

print("🔄 Loading QA system (this may take a few seconds)...")
qa_pipeline = create_qa_pipeline()
print("✅ QA system ready!")

# API route
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Missing query"}), 400
    try:
        answer = qa_pipeline.run(user_query)
        return jsonify({"answer": answer.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == '__main__':
    app.run(debug=True)

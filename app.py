from flask import Flask, request, jsonify
import os
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import TextLoader
import requests

# Flask app
app = Flask(__name__)

# Load Together.ai key from environment
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

# 1. Load and split resumes
def load_documents(data_dir='./resumes'):
    paths = Path(data_dir).glob("*.txt")
    docs = []
    for path in paths:
        loader = TextLoader(str(path))
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# 2. Load vector DB (once)
docs = load_documents()
embeddings = GPT4AllEmbeddings()
vectordb = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 3. Ask endpoint
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve top matching chunks
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Call Together.ai with RAG-style prompt
    prompt = f"""Answer the following question using the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        result = response.json()["choices"][0]["message"]["content"]
        return jsonify({"response": result.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route('/')
def health():
    return "🟢 RAG API (Together.ai) is running."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

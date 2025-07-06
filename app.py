from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Set your Together.ai API key here (or from env)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "your_together_ai_api_key_here")

# Endpoint: GET or POST /ask?query=your-question
@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        query = request.json.get('query', '')
    else:
        query = request.args.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Call Together.ai API
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        data = response.json()
        result = data["choices"][0]["message"]["content"]
        return jsonify({"response": result.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route('/')
def index():
    return "🟢 Together.ai Flask API is running!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

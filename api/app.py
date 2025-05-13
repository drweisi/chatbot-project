import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import json
import redis

load_dotenv()

app = Flask(__name__)
CORS(app)

redis_url = os.getenv("REDIS_URL")
redis_token = os.getenv("REDIS_TOKEN")
r = redis.Redis.from_url(redis_url, password=redis_token, decode_responses=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get("messages", [])
    user_id = data.get("user_id", "default")

    history_key = f"chat_history:{user_id}"
    history_json = r.get(history_key)
    history = json.loads(history_json) if history_json else []

    full_context = history + messages

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=full_context,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return jsonify({"reply": "系统错误，请稍后再试。", "error": str(e)}), 500

    full_context.append({"role": "assistant", "content": reply})
    r.set(history_key, json.dumps(full_context[-20:]), ex=3600)

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)

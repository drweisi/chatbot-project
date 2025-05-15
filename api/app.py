import os
import json
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, make_response
from openai import OpenAI
from dotenv import load_dotenv
import redis

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")  # 例：redis://:密码@HOST:6379/0

client = OpenAI(api_key=OPENAI_API_KEY)

# Redis 初始化
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# 常量设置
HISTORY_LIMIT = 20  # 每用户最多保存20轮
HISTORY_EXPIRE = 3600 * 24  # 聊天历史24小时自动过期

app = Flask(__name__, static_folder="../static", template_folder="../templates")

@app.route("/")
def index():
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid4())
    resp = make_response(render_template("index.html"))
    resp.set_cookie("user_id", user_id, max_age=HISTORY_EXPIRE)
    return resp

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = request.cookies.get("user_id", "anonymous")
    message = data.get("message", "").strip()
    history_key = f"chat_history:{user_id}"

    # 读取历史对话
    try:
        history_json = r.get(history_key)
        history = json.loads(history_json) if history_json else []
    except Exception as e:
        history = []

    # 加入本轮用户消息
    history.append({"role": "user", "content": message})
    # 只保留最近 N 条
    context = history[-HISTORY_LIMIT:]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "你是一个医学知识丰富的智能医生助手，简明、严谨地回答医学相关问题。"}] + context,
            max_tokens=512
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"接口错误：{e}"

    # 加入助手回复
    history.append({"role": "assistant", "content": reply})
    # 只存 N 条历史，自动过期
    try:
        r.set(history_key, json.dumps(history[-HISTORY_LIMIT:]), ex=HISTORY_EXPIRE)
    except Exception as e:
        pass  # Redis 报错时直接跳过，不影响主流程

    return jsonify({"reply": reply, "history": history[-HISTORY_LIMIT:]})

@app.route("/api/history", methods=["GET"])
def get_history():
    user_id = request.cookies.get("user_id", "anonymous")
    history_key = f"chat_history:{user_id}"
    try:
        history_json = r.get(history_key)
        history = json.loads(history_json) if history_json else []
    except Exception as e:
        history = []
    return jsonify({"history": history[-HISTORY_LIMIT:]})

if __name__ == "__main__":
    app.run(debug=True)

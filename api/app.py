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
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

HISTORY_LIMIT = 20
HISTORY_EXPIRE = 3600 * 24
ENABLE_HISTORY = True  # 🔁 只想测单轮对话时改成 False

app = Flask(
    __name__,
    static_folder="../public/static",         # vercel部署建议
    template_folder="../public"
)

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

    # 历史对话（根据开关决定是否启用）
    if ENABLE_HISTORY:
        try:
            history_json = r.get(history_key)
            history = json.loads(history_json) if history_json else []
        except Exception:
            history = []
        history.append({"role": "user", "content": message})
        context = history[-HISTORY_LIMIT:]
        messages = [
            {"role": "system", "content": "你是一个医学知识丰富的智能医生助手，简明、严谨地回答医学相关问题。"}
        ] + context
    else:
        messages = [
            {"role": "system", "content": "你是一个医学知识丰富的智能医生助手，简明、严谨地回答医学相关问题。"},
            {"role": "user", "content": message}
        ]
        history = [{"role": "user", "content": message}]

    print("【DEBUG】messages:", messages)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=512
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"接口错误：{e}"

    # 保存历史
    if ENABLE_HISTORY:
        history.append({"role": "assistant", "content": reply})
        try:
            r.set(history_key, json.dumps(history[-HISTORY_LIMIT:]), ex=HISTORY_EXPIRE)
        except Exception:
            pass

    return jsonify({"reply": reply, "history": history[-HISTORY_LIMIT:]})

# 🔥 清空当前用户历史，调试专用
@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    user_id = request.cookies.get("user_id", "anonymous")
    history_key = f"chat_history:{user_id}"
    try:
        r.delete(history_key)
    except Exception:
        pass
    return jsonify({"msg": "历史已清空"})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template, make_response, send_from_directory, Response, stream_with_context
import os
import base64
import time
import json
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import openai
import cloudinary
import cloudinary.uploader
import cloudinary.api
import redis
import json

# 加载环境变量
load_dotenv()

# 正确设置Flask应用
app = Flask(__name__, 
            static_folder=None,  # 禁用默认静态文件处理
            template_folder='../templates')  

app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# 初始化 OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# 初始化 Redis
redis_url = os.getenv('REDIS_URL')
redis_client = redis.from_url(redis_url) if redis_url else None

# 初始化 Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# 添加静态文件处理路由
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('../public', path)

# 确保用户会话
@app.before_request
def before_request():
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())

    # 将 user_id 存储到请求上下文中供后续使用
    request.user_id = user_id

# 从 Redis 获取会话历史
def get_conversation_history(user_id):
    if not redis_client:
        return []
        
    history_key = f"chat_history:{user_id}"
    history_data = redis_client.get(history_key)
    
    if history_data:
        return json.loads(history_data)
    return []

# 保存会话历史到 Redis
def save_conversation_history(user_id, history):
    if not redis_client:
        return
        
    history_key = f"chat_history:{user_id}"
    # 设置历史记录，有效期 30 天
    redis_client.setex(
        history_key, 
        timedelta(days=30), 
        json.dumps(history)
    )

def should_analyze_image(message_text=None, explicit_request=False):
    """
    使用 GPT-4o 时，总是包含图片进行分析
    """
    return True

def upload_to_cloudinary(image_data):
    """上传图像到 Cloudinary 并返回优化后的 URL"""
    try:
        # 从 Base64 中提取图片数据
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # 上传到 Cloudinary
        result = cloudinary.uploader.upload(
            f"data:image/jpeg;base64,{image_data}",
            folder="medical_chat",  # 创建文件夹
            resource_type="image",
            # 设置图像尺寸和质量限制，减少 token 消耗
            transformation=[
                {'width': 512, 'height': 512, 'crop': 'limit'},
                {'quality': 'auto:good'}
            ],
            # 设置公共 ID 格式
            public_id=f"img_{int(time.time())}_{request.user_id[:8]}",
            # 设置自动删除标签
            tags=["auto_delete_90days"]
        )
        
        # 返回优化后的图片 URL
        return cloudinary.utils.cloudinary_url(
            result['public_id'],
            quality="auto",
            fetch_format="auto"
        )[0]
    except Exception as e:
        print(f"Cloudinary 上传错误: {e}")
        return None

@app.route("/")
def index():
    # 创建用户 cookie 如果不存在
    response = make_response(render_template("index.html"))
    
    if not request.cookies.get('user_id'):
        user_id = str(uuid.uuid4())
        # 设置 cookie 30 天过期
        response.set_cookie('user_id', user_id, max_age=30*24*60*60) 
    
    return response

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()
    image_data = data.get("image")
    analyze_image = data.get("analyze_image", False)  # 用户是否请求图像分析
    stream_response = data.get("stream", True)  # 默认启用流式响应
    
    # 从 Redis 获取历史记录
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
    
    conversation_history = get_conversation_history(user_id)
    
    # 准备发送给 GPT-4O 的消息
    messages = [
        {"role": "system", "content": """
        # 角色定位
        你是专业医疗助手，提供医学知识咨询但不给出确诊。

        # 能力范围
        能分析用户上传的医疗图片，包括:
        1. 医学报告(血液检查、影像报告等)
        2. 医疗影像(X光、CT、MRI等)
        3. 皮肤情况
        4. 药品信息

        # 行为准则
        分析图片时:
        - 提供详细专业分析，指出关键数据和异常值
        - 明确声明不构成医疗诊断
        - 严重异常时建议咨询医生
        - 不假装看到未提供的信息
        - 保持专业、准确和谨慎

        # 图片处理流程
        处理用户图片时遵循以下步骤:
        1. 判断是否医疗相关(症状、报告、药物等为相关；风景、食物等为非相关)
        2. 非医疗图片:
        - 告知用户"您上传的图片不是医疗相关内容，请上传与医疗问题相关的图片"
        - 不分析非医疗图片内容
        - 不提供任何医疗解读
        3. 医疗图片:
        - 谨慎分析提供专业见解
        - 分析用户上传的图片时，请尽量提供详细而专业的分析，指出关键数据和异常值，并解释其潜在意义。
        - 强调仅提供参考信息
        - 建议咨询专业医生

        # 沟通风格
        - 专业但平易近人
        - 使用准确医学术语并提供必要解释
        - 回答清晰、结构化

        # 安全与限制
        严格遵循处理流程，确保不对非医疗图片进行医疗分析。

        # 其他指导
        [预留位置，可添加额外指令]
        """},
    ]
    
    # 添加历史对话
    for exchange in conversation_history:
        messages.append(exchange)
    
    # 准备用户消息内容
    user_content = []
    
    # 添加文本消息
    if message:
        user_content.append({"type": "text", "text": message})
    
    # 如果有图片，判断是否需要分析
    image_url = None
    if image_data:
        # 先上传图片到 Cloudinary，无论是否需要分析
        image_url = upload_to_cloudinary(image_data)
        
        # 判断是否需要将图片加入分析
        if should_analyze_image(message, analyze_image) and image_url:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            
            # 如果用户没有提供文字说明，添加默认提示
            if not message:
                user_content = [
                    {"type": "text", "text": "请帮我分析这张图片，告诉我这是否是医疗相关内容以及你看到了什么。"},
                    user_content[0]  # 图片内容
                ]
    
    # 只有当有内容可以发送时才继续
    if not user_content:
        return jsonify({"response": "请提供文字消息或图片。"})
    
    # 添加用户消息到发送队列
    messages.append({"role": "user", "content": user_content})
    
    try:
        # 根据请求选择是流式响应还是普通响应
        if stream_response:
            return stream_chat_response(messages, user_id, conversation_history, message, image_url)
        else:
            return normal_chat_response(messages, user_id, conversation_history, message, image_url)
    except Exception as e:
        print(f"OpenAI API 错误: {e}")
        return jsonify({"response": f"抱歉，发生了错误: {str(e)}"})

def stream_chat_response(messages, user_id, conversation_history, message, image_url):
    """处理流式响应"""
    def generate():
        full_response = ""
        # 调用 OpenAI 流式 API
        try:
            stream = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
                stream=True  # 启用流式输出
            )
            
            # 逐块发送流数据
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # 包装成 JSON 格式发送
                    yield f"data: {json.dumps({'chunk': content})}\n\n"
            
            # 全部内容发送完毕，更新会话历史
            if image_url:
                conversation_history.append({
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": message if message else "请分析这张图片"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                conversation_history.append({
                    "role": "user", 
                    "content": message
                })
                
            # 保存助手回复到历史
            conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            # 限制历史长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
            # 更新 Redis 中的会话状态
            save_conversation_history(user_id, conversation_history)
            
            # 发送完成信号
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            print(f"流式响应错误: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    # 返回流式响应
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )

def normal_chat_response(messages, user_id, conversation_history, message, image_url):
    """处理普通响应"""
    # 调用 OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
        max_tokens=800
    )
    
    # 获取助手回复
    assistant_response = response.choices[0].message.content
    
    # 更新会话历史
    # 保存用户消息到历史
    if image_url:
        conversation_history.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": message if message else "请分析这张图片"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })
    else:
        conversation_history.append({
            "role": "user", 
            "content": message
        })
        
    # 保存助手回复到历史
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    # 限制历史长度以节省 token 和内存
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    # 更新 Redis 中的会话状态
    save_conversation_history(user_id, conversation_history)
    
    # 返回响应
    return jsonify({
        "response": assistant_response
    })

@app.route("/api/clear", methods=["POST"])
def clear_conversation():
    # 获取用户 ID
    user_id = request.cookies.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "找不到用户会话"})
        
    # 清除 Redis 中的会话历史
    if redis_client:
        redis_client.delete(f"chat_history:{user_id}")
        
    return jsonify({"status": "success", "message": "会话已清除"})

@app.route("/api/cleanup", methods=["POST"])
def cleanup_endpoint():
    if request.headers.get('Authorization') != f"Bearer {os.getenv('CRON_SECRET')}":
        return jsonify({"status": "error", "message": "未授权访问"}), 401
        
    try:
        ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime('%d-%m-%Y')
        result = cloudinary.api.delete_resources_by_tag(
            "auto_delete_90days",
            type="upload",
            resource_type="image",
            created_at_lte=ninety_days_ago
        )
        return jsonify({"status": "success", "message": "清理完成", "details": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

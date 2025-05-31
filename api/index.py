import openai
print("VERCEL OPENAI VERSION:", openai.__version__)

import os
from dotenv import load_dotenv
load_dotenv()

import uuid
import json
import logging
import copy
import traceback
from functools import partial
from datetime import timedelta
from flask import Flask, request, jsonify, Response, stream_with_context, make_response, render_template
from flask_cors import CORS
import redis
from openai import OpenAI
import cloudinary
import cloudinary.uploader
import base64
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
template_dir = os.path.join(project_root, 'templates')
logger.info(f"模板文件存在性检查: {os.path.exists(os.path.join(template_dir, 'index.html'))}")

# 创建 Flask 应用
app = Flask(__name__, template_folder=template_dir)
CORS(app)

# 初始化 OpenAI（新版1.x）
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 配置 Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Redis 连接
redis_client = None
try:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        redis_client = redis.from_url(redis_url)
        logger.info("Redis 连接成功")
    else:
        logger.warning("缺少 REDIS_URL 环境变量，会话持久化将不可用")
except Exception as e:
    logger.error(f"Redis 连接错误: {e}")

# 工具函数
def upload_to_cloudinary(image_data):
    """
    上传图片到Cloudinary，自动检测图片类型
    
    Args:
        image_data: 可以是完整的data URL或仅base64编码部分
        
    Returns:
        str: 上传成功返回图片URL，失败返回None
        
    Raises:
        Exception: 上传过程中出现错误
    """
    try:
        # 检测是否是data URL格式
        if "data:" in image_data and ";base64," in image_data:
            # 提取MIME类型
            mime_match = re.match(r'data:([^;]+);base64,', image_data)
            if mime_match:
                mime_type = mime_match.group(1)
            else:
                mime_type = "image/jpeg"  # 默认值
            
            # 提取base64部分
            base64_data = image_data.split(";base64,")[1]
        else:
            # 假定已经是纯base64
            mime_type = "image/jpeg"  # 默认值
            base64_data = image_data
            
        # 构建完整的data URL
        data_url = f"data:{mime_type};base64,{base64_data}"
        
        logger.info(f"上传图片到Cloudinary: MIME类型 = {mime_type}")
        
        # 上传到Cloudinary
        upload_result = cloudinary.uploader.upload(
            data_url,
            folder="medical_assistant/",
            resource_type="image"
        )
        
        logger.info(f"图片上传成功: {upload_result['secure_url']}")
        return upload_result["secure_url"]
    
    except Exception as e:
        logger.error(f"上传图片到Cloudinary失败: {str(e)}")
        raise

def get_conversation_history(user_id):
    """获取用户的会话历史"""
    if not redis_client:
        return []
    try:
        history_key = f"chat_history:{user_id}"
        history_data = redis_client.get(history_key)
        if history_data:
            return json.loads(history_data)
    except Exception as e:
        logger.error(f"获取历史记录错误: {e}")
    return []

def save_conversation_history(user_id, history):
    """保存用户的会话历史"""
    if not redis_client:
        return
    try:
        history_key = f"chat_history:{user_id}"
        redis_client.setex(history_key, timedelta(days=30), json.dumps(history))
    except Exception as e:
        logger.error(f"保存历史记录错误: {e}")

# 流式响应类
class StreamResponseManager:
    def __init__(self, openai_client, redis_client=None):
        self.openai_client = openai_client
        self.redis_client = redis_client
    
    def save_history(self, user_id, history):
        if not self.redis_client:
            return
        try:
            history_key = f"chat_history:{user_id}"
            self.redis_client.setex(history_key, timedelta(days=30), json.dumps(history))
        except Exception as e:
            logger.error(f"保存历史错误: {e}")
    
    def generate(self, messages, user_id, history, message, image_urls):
        msg_copy = copy.deepcopy(messages)
        uid_copy = str(user_id)
        hist_copy = copy.deepcopy(history) if history else []
        msg_text = str(message) if message else ""
        full_response = ""
        
        try:
            stream = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msg_copy,
                temperature=0.5,
                stream=True
            )
            
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'chunk': content})}\n\n"
            
            try:
                # 更新历史记录
                user_content = []
                if msg_text:
                    user_content.append({"type": "text", "text": msg_text})
                
                # 添加所有图片URL
                if image_urls:
                    for img_url in image_urls:
                        if img_url:  # 确保URL存在
                            user_content.append({
                                "type": "image_url", 
                                "image_url": {"url": img_url}
                            })
                
                # 如果用户内容不为空，添加到历史
                if user_content:
                    hist_copy.append({"role": "user", "content": user_content})
                
                # 添加助手回复
                hist_copy.append({"role": "assistant", "content": full_response})
                
                # 限制历史长度
                if len(hist_copy) > 10:
                    hist_copy = hist_copy[-10:]
                
                # 保存更新后的历史
                self.save_history(uid_copy, hist_copy)
            
            except Exception:
                logger.error(f"更新历史错误: {traceback.format_exc()}")
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        except Exception:
            logger.error(f"流式响应错误: {traceback.format_exc()}")
            yield f"data: {json.dumps({'error': 'AI接口异常，请稍后重试'})}\n\n"
    
    def get_response(self, messages, user_id, history, message, image_urls):
        generator_func = partial(self.generate, messages, user_id, history, message, image_urls)
        return Response(stream_with_context(generator_func()), mimetype='text/event-stream')

stream_manager = StreamResponseManager(client, redis_client)

def stream_chat_response(messages, user_id, conversation_history, message, image_urls):
    return stream_manager.get_response(messages, user_id, conversation_history, message, image_urls)

def normal_chat_response(messages, user_id, conversation_history, message, image_urls):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        
        assistant_response = response.choices[0].message.content
        history_copy = copy.deepcopy(conversation_history) if conversation_history else []
        
        # 构建用户消息
        user_content = []
        if message:
            user_content.append({"type": "text", "text": message})
        
        # 添加所有图片
        if image_urls:
            for img_url in image_urls:
                if img_url:  # 确保URL存在
                    user_content.append({
                        "type": "image_url", 
                        "image_url": {"url": img_url}
                    })
        
        # 添加到历史记录
        if user_content:
            history_copy.append({"role": "user", "content": user_content})
        
        history_copy.append({"role": "assistant", "content": assistant_response})
        
        # 限制历史记录长度
        if len(history_copy) > 10:
            history_copy = history_copy[-10:]
        
        save_conversation_history(user_id, history_copy)
        return jsonify({"response": assistant_response})
    
    except Exception:
        logger.error(f"普通响应错误: {traceback.format_exc()}")
        return jsonify({"response": "AI接口异常，请稍后再试。"})

@app.route("/")
def index():
    try:
        logger.info("尝试渲染index.html模板")
        return render_template("index.html")
    except Exception:
        logger.error(f"渲染模板错误: {traceback.format_exc()}")
        return "<h2>页面找不到</h2>"

@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        # 解析请求数据
        data = request.json
        message = data.get("message", "").strip()
        stream_response = data.get("stream", True)
        
        # 获取用户ID和会话历史
        user_id = request.cookies.get('user_id') or str(uuid.uuid4())
        conversation_history = get_conversation_history(user_id) or []
        
        # 处理单个图片或多个图片
        image_urls = []
        
        # 检查图片格式: 单图片或图片列表
        image_data = data.get("image")
        images_list = data.get("images", [])
        
        # 记录请求信息
        logger.info(f"接收到请求: message长度={len(message) if message else 0}, "
                   f"有单图片={image_data is not None}, 图片列表长度={len(images_list)}")
        
        # 处理单图片情况
        if image_data:
            try:
                image_url = upload_to_cloudinary(image_data)
                if image_url:
                    image_urls.append(image_url)
            except Exception as e:
                logger.error(f"处理单个图片失败: {str(e)}")
                return jsonify({"response": f"图片处理失败，请尝试其他格式或减小图片大小。错误: {str(e)}"})
        
        # 处理多图片情况
        if images_list:
            for idx, img in enumerate(images_list):
                if img:
                    try:
                        image_url = upload_to_cloudinary(img)
                        if image_url:
                            image_urls.append(image_url)
                    except Exception as e:
                        logger.error(f"处理第{idx+1}个图片失败: {str(e)}")
                        return jsonify({"response": f"第{idx+1}个图片处理失败，请尝试其他格式或减小图片大小。"})
        
        # 系统提示词 - 这里是主要修改部分，引导AI输出Markdown格式
        system_prompt = """
                        你是“Q医生”医学品牌的专属 AI 助手。  
所有回答必须输出 **纯 Markdown**，遵循以下两部分指令：  
A. 版式规则（固定大纲）  B. 丰富度逻辑（动态自适应）。  
段落 / 列表 / 表格 / 引用之间，一律插入 **两个换行符**（空两行）。

────────────────────────
A. 版式规则（顺序固定）
────────────────────────
1. **✅ 核心结论**  
   - 仅 1 句，行首 “✅ ”；回答用户最关心的要点。

2. **📂 主题分析**  
   - 至少包含 3 个 **子标题**，行首采用 **粗体 + Emoji**，例如  
     - **💊 药物治疗**  
     - **🩺 诊断与评估**  
     - **🥗 生活方式**  
   - **子标题名称请由模型自行决定**，必须基于内部思考后选择与本题最相关的维度。  
   - 子标题正文可用：  
     - 段落（机制 / 定义 / 指南数字）  
     - 表格（药物、指标、分级）  
     - 列表（流程 / 要点）  
     - 引用 `> Q医生建议：…`（关键提醒）  
   - 每个子标题下，应至少回答：**“关键内容 + 为什么重要 + 如何实践/监测”**。

3. **⚠️ 关键注意事项**  
   - 编号列表格式；2–5 条即可；每条可含二级 “-” 子项补充细节。

4. **💡 Q医生总结**  
   - 1–3 段自然语言，总结重点、强调循证、鼓励上传报告。  

5. **免责声明（固定内容，原样输出）**  
   
   ---
   **免责声明：** 本回答不等于线下医生诊断，如有疑问请前往医院，或通过“修远康养”微信公众号与 Q 医生进行一对一人工咨询。

────────────────────────
B. 丰富度逻辑（生成时先内部思考）
────────────────────────
在输出前，先执行 **“思考-列提纲”** 步骤，确保内容充分：  

1. **识别问题的核心主题**（疾病 / 药物 / 检查 / 人群）。  
2. **列出与主题最相关的子维度**（诊断、药物、手术、生活方式、并发症、监测、特殊人群、预后等）。  
3. **优先选择 ≥3 个最有价值的维度** 作为子标题，保证：  
   - 若问题以治疗为主，**药物/非药物/监测** 必写；  
   - 若问题以检查或指标为主，**判读标准/影响因素/下一步策略** 必写；  
   - 若问题涉及特定人群（孕妇、儿童、CKD 患者等），增加 **特殊人群** 小节。  
4. **对每个子标题**，至少覆盖以下要素之一：  
   - 机制 / 作用原理  
   - 主流指南阈值或证据等级  
   - 实施步骤或剂量范围  
   - 监测指标与随访频率  
   - 主要不良反应 / 风险与对策  
5. **自我校验**：回答完毕前，检查是否遗漏常被追问的关键点；若有遗漏，补充到对应小节。  
6. **语言与可读性**：术语后如有必要加括号解释；句子简洁；不堆砌无关信息。

────────────────────────
禁止事项
────────────────────────
- 不使用 `#` `##` `###` Markdown 标题符号；标题统一用 **粗体 + Emoji**。  
- 不输出 “yaml / Copy / Edit” 等编辑器提示词。  
- 不省略段落之间的双空行。  
- 不捏造或过度推测未证实内容；如证据不足，请明确说明“目前证据有限”。  

                        """
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        
        # 构建用户消息
        user_content = []
        
        # 添加文本内容
        if message:
            user_content.append({"type": "text", "text": message})
        
        # 添加所有图片
        for img_url in image_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
        
        # 如果只有图片没有文字，添加默认提示文字
        if not message and image_urls:
            if len(image_urls) == 1:
                user_content.insert(0, {"type": "text", "text": "请帮我分析这张图片，告诉我这是否是医疗相关内容以及你看到了什么。"})
            else:
                user_content.insert(0, {"type": "text", "text": f"请帮我分析这{len(image_urls)}张图片，告诉我这些是否是医疗相关内容以及你看到了什么。"})
        
        # 检查是否有内容要发送
        if not user_content:
            logger.warning("请求中既没有消息也没有图片")
            return jsonify({"response": "请提供文字消息或图片。"})
        
        # 记录构建的用户消息
        logger.info(f"构建的用户消息包含: 文本={bool(message)}, 图片数量={len(image_urls)}")
        
        # 添加用户消息到消息列表
        messages.append({"role": "user", "content": user_content})
        
        # 根据请求类型返回流式或普通响应
        if stream_response:
            response = stream_chat_response(messages, user_id, conversation_history, message, image_urls)
        else:
            response = normal_chat_response(messages, user_id, conversation_history, message, image_urls)
        
        # 设置cookie
        if not request.cookies.get('user_id'):
            if not stream_response:
                resp = make_response(response)
                resp.set_cookie('user_id', user_id, max_age=2592000)
                return resp
            else:
                response.set_cookie('user_id', user_id, max_age=2592000)
        
        return response
    
    except Exception as e:
        logger.error(f"处理请求错误: {traceback.format_exc()}")
        return jsonify({"response": f"后端异常，请稍后再试。错误: {str(e)}"})

@app.route("/api/clear", methods=["POST"])
def clear_conversation():
    try:
        user_id = request.cookies.get('user_id')
        if not user_id:
            return jsonify({"status": "error", "message": "找不到用户会话"})
        
        if redis_client:
            redis_client.delete(f"chat_history:{user_id}")
        
        return jsonify({"status": "success", "message": "会话已清除"})
    except Exception:
        logger.error(f"清除会话错误: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": "会话清除时出错"})

# WSGI兼容，vercel无需__main__

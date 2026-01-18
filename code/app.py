# -*- coding: utf-8 -*-
"""
智能问答系统 - 后端主程序
最终版：修复数组布尔判断错误、接入豆包API、完整可运行
"""
import os
import sys
import time
import requests
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

# ===================== 解决模块导入路径问题 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 导入自定义模块（确保路径正确）
try:
    from nlp_model_loader import load_nlp_model, infer_nlp_model
    from utils.result_utils import format_result_simple, save_result_simple
    from utils.data_utils import preprocess_text
except ImportError as e:
    print(f"⚠️  自定义模块导入失败：{e}")
    print(f"⚠️  请确保nlp_model_loader.py和utils目录在{BASE_DIR}目录下")
    
    # 模拟函数（修复数组布尔判断问题）
    def load_nlp_model(path):
        return None
    
    def infer_nlp_model(model, text, task):
        # 模拟返回值，避免多维数组布尔判断
        result = f"模拟{task}结果：{text}"
        return result
    
    def format_result_simple(input_text, task_type, result_data):
        # 修复：先判断是否为数组，再处理
        content = result_data["content"]
        if isinstance(content, np.ndarray):
            # 数组转字符串，避免布尔判断错误
            content = content.tolist() if content.size > 1 else content.item()
        return str(content)
    
    def save_result_simple(input_text, task_type, result_data, cost_time):
        return ""
    
    def preprocess_text(text):
        # 预处理返回字符串，避免返回数组
        return str(text).strip()

# ===================== 豆包API配置（已填入你的API Key） =====================
DOUBAO_CONFIG = {
    "API_KEY": "41a24a92-863e-4522-9c54-4ef608d096c2",
    "API_URL": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "MODEL": "ep-20251228194024-9ql9j"
}

# ===================== 初始化Flask配置 =====================
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")
STATIC_DIR = os.path.join(BASE_DIR, "../static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
    static_url_path=""
)

# 全局配置常量
CONFIG = {
    "MODEL_PATH": os.path.join(BASE_DIR, "../model/nlp_multi_task.h5"),
    "NLP_MODEL": None,
    "SUPPORTED_TASKS": ["chat", "classify", "sentiment", "translate"]
}

model_loaded = False

# ===================== 模型初始化 =====================
def init_model():
    """初始化加载本地NLP模型（修复异常处理）"""
    global model_loaded
    if not model_loaded:
        if os.path.exists(CONFIG["MODEL_PATH"]):
            try:
                CONFIG["NLP_MODEL"] = load_nlp_model(CONFIG["MODEL_PATH"])
                print(f"✅ 成功加载模型：{CONFIG['MODEL_PATH']}")
                model_loaded = True
            except Exception as e:
                print(f"❌ 模型加载失败：{str(e)}")
        else:
            print(f"⚠️  未找到模型文件：{CONFIG['MODEL_PATH']}")
            print(f"⚠️  本地模型功能将不可用，仅能使用智能问答功能")

@app.before_request
def before_request():
    init_model()

# ===================== 豆包API调用函数 =====================
def call_doubao_api(input_text):
    """调用豆包API获取真实回复"""
    payload = {
        "model": DOUBAO_CONFIG["MODEL"],
        "messages": [{"role": "user", "content": input_text}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_CONFIG['API_KEY']}"
    }
    
    try:
        response = requests.post(
            DOUBAO_CONFIG["API_URL"],
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        api_result = response.json()
        return api_result["choices"][0]["message"]["content"]
    
    except requests.exceptions.Timeout:
        return "API调用超时，请重试"
    except requests.exceptions.HTTPError as e:
        return f"API调用失败（HTTP错误）：{str(e)}"
    except Exception as e:
        return f"API调用失败：{str(e)}"

# ===================== 核心工具函数（修复数组判断） =====================
def safe_array_check(value):
    """安全检查数组/张量的布尔值，修复歧义性错误"""
    if isinstance(value, np.ndarray):
        # 空数组返回False，非空数组判断是否有有效值
        return value.size > 0 and np.any(value)
    elif hasattr(value, 'numpy'):  # 处理TensorFlow张量
        value = value.numpy()
        return value.size > 0 and np.any(value)
    else:
        # 普通类型直接判断
        return bool(value)

# ===================== 路由函数 =====================
@app.route("/")
def index():
    """首页：渲染前端页面"""
    js_path = os.path.join(STATIC_DIR, "js/main.js")
    css_path = os.path.join(STATIC_DIR, "css/main.css")
    print(f"\n📌 静态文件检查：")
    print(f"   main.js路径：{js_path} | 存在：{os.path.exists(js_path)}")
    print(f"   main.css路径：{css_path} | 存在：{os.path.exists(css_path)}")
    
    return render_template(
        "index.html",
        supported_tasks=CONFIG["SUPPORTED_TASKS"]
    )

@app.route("/api/process", methods=["POST"])
def process_text():
    """处理文本请求的核心接口（修复数组布尔判断错误）"""
    try:
        # 1. 获取请求参数
        request_data = request.get_json()
        input_text = request_data.get("text", "").strip()
        task_type = request_data.get("task_type", "chat")

        # 2. 参数校验
        if not input_text:
            return jsonify({
                "code": 400,
                "msg": "输入文本不能为空",
                "data": {}
            })
        if task_type not in CONFIG["SUPPORTED_TASKS"]:
            return jsonify({
                "code": 400,
                "msg": f"不支持的任务类型：{task_type}，仅支持{CONFIG['SUPPORTED_TASKS']}",
                "data": {}
            })

        # 3. 记录处理开始时间
        start_time = time.time()

        # 4. 处理不同任务类型
        result_data = {
            "type": task_type,
            "source": "",
            "content": ""
        }

        # 4.1 智能问答（调用豆包API）
        if task_type == "chat":
            result_data["source"] = "豆包API"
            real_reply = call_doubao_api(input_text)
            result_data["content"] = f"📝 你的问题：{input_text}\n\n🤖 豆包回复：\n{real_reply}"
        
        # 4.2 本地模型推理（修复数组判断逻辑）
        else:
            if CONFIG["NLP_MODEL"] is None:
                result_data["source"] = "本地模型（未加载）"
                result_data["content"] = f"⚠️ 本地{task_type}模型未加载\n模拟结果：{input_text}"
            else:
                try:
                    # 文本预处理（确保返回字符串）
                    processed_text = preprocess_text(input_text)
                    processed_text = str(processed_text) if safe_array_check(processed_text) else ""
                    
                    # 模型推理
                    infer_result = infer_nlp_model(CONFIG["NLP_MODEL"], processed_text, task_type)
                    
                    # 安全处理推理结果（数组转字符串）
                    if isinstance(infer_result, np.ndarray):
                        # 多维数组转列表，一维数组转字符串
                        infer_result = infer_result.tolist() if infer_result.ndim > 1 else ", ".join(map(str, infer_result))
                    elif hasattr(infer_result, 'numpy'):
                        infer_result = infer_result.numpy().tolist()
                    
                    # 格式化结果（避免布尔判断错误）
                    format_input = {
                        "content": str(infer_result) if safe_array_check(infer_result) else "无结果",
                        "source": "本地NLP模型"
                    }
                    result_data["content"] = format_result_simple(input_text, task_type, format_input)
                    result_data["source"] = "本地NLP模型"
                    
                except Exception as e:
                    result_data["source"] = "本地模型（推理失败）"
                    result_data["content"] = f"⚠️ 本地{task_type}模型推理出错：{str(e)}\n模拟结果：{input_text}"

        # 5. 计算处理耗时并保存结果
        cost_time = time.time() - start_time
        save_result_simple(input_text, task_type, result_data, cost_time)

        # 6. 返回响应
        return jsonify({
            "code": 200,
            "msg": "处理成功",
            "data": result_data
        })

    except Exception as e:
        error_msg = f"处理失败：{str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            "code": 500,
            "msg": error_msg,
            "data": {}
        })

# ===================== 启动程序 =====================
if __name__ == "__main__":
    # 自动创建所有必要目录
    required_dirs = [
        os.path.join(BASE_DIR, "../data/ids"),
        os.path.join(BASE_DIR, "../model"),
        os.path.join(BASE_DIR, "../tmp/results"),
        TEMPLATE_DIR,
        STATIC_DIR,
        os.path.join(STATIC_DIR, "css"),
        os.path.join(STATIC_DIR, "js")
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 确保目录存在：{dir_path}")

    # 启动信息
    print("\n🚀 智能问答系统启动成功！")
    print(f"📌 支持的任务类型：{CONFIG['SUPPORTED_TASKS']}")
    print(f"📌 模型文件路径：{CONFIG['MODEL_PATH']}")
    print(f"📌 模板文件路径：{TEMPLATE_DIR}")
    print(f"📌 静态文件路径：{STATIC_DIR}")
    print("🌐 访问地址：http://127.0.0.1:5000")
    
    # 启动Flask服务
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
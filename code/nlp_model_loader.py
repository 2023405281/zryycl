import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import json

# 全局配置（可根据你的模型实际情况调整）
MODEL_CONFIG = {
    # 文本预处理配置（需与模型训练时保持一致）
    "MAX_SEQ_LEN": 128,          # 文本最大长度
    "VOCAB_SIZE": 10000,         # 词汇表大小
    "EMBEDDING_DIM": 128,        # 嵌入层维度
    # 任务相关配置
    "CLASSIFY_LABELS": ["科技", "教育", "娱乐", "财经", "体育"],  # 文本分类标签
    "SENTIMENT_LABELS": ["负面", "中性", "正面"],                # 情感分析标签
    "TRANSLATE_MAX_LEN": 64,     # 翻译文本最大长度
    # 默认填充/未知token
    "PAD_TOKEN": 0,
    "UNK_TOKEN": 1
}

def load_nlp_model(model_path: str) -> dict:
    """
    加载训练好的.h5格式NLP模型
    :param model_path: .h5模型文件路径
    :return: 包含模型实例和配置的字典，加载失败返回None
    """
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在：{model_path}")
            return None
        
        # 禁用TensorFlow不必要的日志
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        # 加载.h5模型（兼容自定义层/函数）
        model = load_model(
            model_path,
            compile=False,  # 推理阶段无需编译
            custom_objects=None  # 若有自定义层，需在此指定
        )
        
        print(f"✅ 成功加载模型：{model_path}")
        print(f"📌 模型输入形状：{model.input_shape}")
        print(f"📌 模型输出形状：{model.output_shape}")
        
        # 返回模型实例和配置
        return {
            "model": model,
            "config": MODEL_CONFIG,
            "status": "loaded"
        }
    
    except Exception as e:
        print(f"❌ 加载模型失败：{str(e)}")
        return None

def preprocess_input(text: str, config: dict) -> np.ndarray:
    """
    文本预处理（对接utils.data_utils）
    :param text: 原始输入文本
    :param config: 模型配置字典
    :return: 模型可接受的张量输入（shape: (1, MAX_SEQ_LEN)）
    """
    try:
        from utils.data_utils import preprocess_text
        return preprocess_text(text, config)
    except Exception as e:
        print(f"❌ 调用utils预处理失败，使用备用逻辑：{str(e)}")
        # 备用逻辑（兼容无utils的情况）
        text = text[:config["MAX_SEQ_LEN"]]
        char_ids = [ord(c) % config["VOCAB_SIZE"] for c in text]
        char_ids += [config["PAD_TOKEN"]] * (config["MAX_SEQ_LEN"] - len(char_ids))
        return np.array([char_ids[:config["MAX_SEQ_LEN"]]], dtype=np.int32)

def infer_nlp_model(model_wrapper: dict, text: str, task_type: str) -> str:
    """
    调用模型进行推理，适配不同任务类型
    :param model_wrapper: load_nlp_model返回的模型包装字典
    :param text: 输入文本
    :param task_type: 任务类型（classify/sentiment/translate）
    :return: 人类可读的推理结果字符串
    """
    # 校验输入
    if not model_wrapper or model_wrapper["status"] != "loaded":
        return "模型未加载，无法进行推理"
    if not text:
        return "输入文本不能为空"
    if task_type not in ["classify", "sentiment", "translate"]:
        return f"不支持的任务类型：{task_type}，仅支持classify/sentiment/translate"
    
    try:
        # 1. 文本预处理
        input_tensor = preprocess_input(text, model_wrapper["config"])
        if input_tensor is None:
            return "文本预处理失败，无法推理"
        
        # 2. 模型推理
        model = model_wrapper["model"]
        predictions = model.predict(input_tensor, verbose=0)
        
        # 3. 根据任务类型解析预测结果
        if task_type == "classify":
            # 文本分类：取概率最大的标签
            pred_idx = np.argmax(predictions[0])
            pred_label = model_wrapper["config"]["CLASSIFY_LABELS"][pred_idx]
            pred_prob = round(float(np.max(predictions[0])), 4)
            return f"分类结果：{pred_label}（置信度：{pred_prob}）\n所有类别概率：{dict(zip(model_wrapper['config']['CLASSIFY_LABELS'], predictions[0].round(4)))}"
        
        elif task_type == "sentiment":
            # 情感分析：取概率最大的标签
            pred_idx = np.argmax(predictions[0])
            pred_label = model_wrapper["config"]["SENTIMENT_LABELS"][pred_idx]
            pred_prob = round(float(np.max(predictions[0])), 4)
            return f"情感分析结果：{pred_label}（置信度：{pred_prob}）\n负面概率：{predictions[0][0].round(4)} | 中性概率：{predictions[0][1].round(4)} | 正面概率：{predictions[0][2].round(4)}"
        
        elif task_type == "translate":
            # 机器翻译：示例逻辑（需根据你的模型输出调整）
            # 此处仅为占位，需替换为实际的翻译ID转文本逻辑
            pred_ids = predictions[0].argmax(axis=-1)[:model_wrapper["config"]["TRANSLATE_MAX_LEN"]]
            # 示例：ID转回字符（实际需加载翻译词汇表）
            translate_text = "".join([chr(int(id) % 65535) for id in pred_ids if id != model_wrapper["config"]["PAD_TOKEN"]])
            return f"翻译结果：{translate_text.strip()}"
    
    except Exception as e:
        return f"推理失败：{str(e)}"

# 测试代码（单独运行该文件验证模型加载和推理）
if __name__ == "__main__":
    # 测试模型加载
    TEST_MODEL_PATH = "../model/nlp_multi_task.h5"  # 模型路径
    model_wrapper = load_nlp_model(TEST_MODEL_PATH)
    
    if model_wrapper:
        # 测试不同任务的推理
        TEST_TEXT = "人工智能技术的发展给教育行业带来了巨大变革"
        
        print("\n===== 文本分类测试 =====")
        classify_result = infer_nlp_model(model_wrapper, TEST_TEXT, "classify")
        print(classify_result)
        
        print("\n===== 情感分析测试 =====")
        sentiment_result = infer_nlp_model(model_wrapper, TEST_TEXT, "sentiment")
        print(sentiment_result)
        
        print("\n===== 机器翻译测试 =====")
        translate_result = infer_nlp_model(model_wrapper, TEST_TEXT, "translate")
        print(translate_result)
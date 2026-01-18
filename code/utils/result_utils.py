# -*- coding: utf-8 -*-
"""
智能问答系统 - 结果处理工具模块
负责结果格式化、整合、可视化、日志记录等
"""
import json
import time
import os
import re
from datetime import datetime

# 结果保存配置
RESULT_CONFIG = {
    "SAVE_DIR": "../../tmp/results",  # 结果保存目录
    "LOG_FILE": "../../tmp/result_logs.txt",  # 结果日志文件
    "SUPPORTED_FORMATS": ["text", "json", "markdown"]  # 支持的输出格式
}

class ResultProcessor:
    """结果处理工具类：格式化、保存、日志、可视化"""
    def __init__(self, config: dict = None):
        self.config = config or RESULT_CONFIG
        # 创建结果保存目录（若不存在）
        os.makedirs(self.config["SAVE_DIR"], exist_ok=True)
        # 初始化日志文件
        self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件，添加表头（若不存在）"""
        if not os.path.exists(self.config["LOG_FILE"]):
            with open(self.config["LOG_FILE"], "w", encoding="utf-8") as f:
                header = "时间\t任务类型\t输入文本\t结果来源\t处理结果\t耗时(秒)\n"
                f.write(header)

    def format_result(self, 
                      input_text: str,
                      task_type: str,
                      result_data: dict,
                      output_format: str = "text") -> str:
        """
        统一格式化结果（适配前端展示）
        :param input_text: 用户输入文本
        :param task_type: 任务类型（chat/classify/sentiment/translate）
        :param result_data: 原始结果数据（包含content/source）
        :param output_format: 输出格式（text/json/markdown）
        :return: 格式化后的结果字符串
        """
        try:
            if output_format not in self.config["SUPPORTED_FORMATS"]:
                raise ValueError(f"不支持的输出格式：{output_format}，仅支持{self.config['SUPPORTED_FORMATS']}")
            
            # 基础结果信息
            base_info = {
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "任务类型": self._get_task_name(task_type),
                "输入文本": input_text[:50] + "..." if len(input_text) > 50 else input_text,
                "结果来源": result_data.get("source", "未知"),
                "处理结果": result_data.get("content", "无结果")
            }

            # 1. 纯文本格式（默认，适配前端展示）
            if output_format == "text":
                return base_info["处理结果"]
            
            # 2. JSON格式（便于保存/传输）
            elif output_format == "json":
                return json.dumps(base_info, ensure_ascii=False, indent=2)
            
            # 3. Markdown格式（便于可视化展示）
            elif output_format == "markdown":
                md_template = f"""
### {base_info['任务类型']}结果
- **处理时间**：{base_info['时间']}
- **输入文本**：{base_info['输入文本']}
- **结果来源**：{base_info['结果来源']}
- **处理结果**：
{base_info['处理结果']}
                """
                return md_template.strip()
        
        except Exception as e:
            error_msg = f"结果格式化失败：{str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def save_result(self, 
                    input_text: str,
                    task_type: str,
                    result_data: dict,
                    cost_time: float,
                    save_format: str = "json") -> str:
        """
        保存结果到文件
        :param input_text: 用户输入文本
        :param task_type: 任务类型
        :param result_data: 结果数据
        :param cost_time: 处理耗时（秒）
        :param save_format: 保存格式（text/json/markdown）
        :return: 保存的文件路径
        """
        try:
            # 生成唯一文件名（时间戳+任务类型）
            timestamp = int(time.time())
            filename = f"{timestamp}_{task_type}_result.{save_format}"
            save_path = os.path.join(self.config["SAVE_DIR"], filename)
            
            # 格式化结果
            formatted_result = self.format_result(
                input_text=input_text,
                task_type=task_type,
                result_data=result_data,
                output_format=save_format
            )
            
            # 保存到文件
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(formatted_result)
            
            # 记录日志
            self._log_result(input_text, task_type, result_data, cost_time)
            
            print(f"✅ 结果已保存至：{save_path}")
            return save_path
        
        except Exception as e:
            print(f"❌ 保存结果失败：{str(e)}")
            return ""

    def _log_result(self, input_text: str, task_type: str, result_data: dict, cost_time: float):
        """记录结果到日志文件"""
        try:
            # 格式化日志行（制表符分隔）
            log_line = (
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t"
                f"{self._get_task_name(task_type)}\t"
                f"{input_text[:100].replace('\t', ' ')}\t"  # 替换制表符避免格式错乱
                f"{result_data.get('source', '未知')}\t"
                f"{result_data.get('content', '无结果')[:200].replace('\t', ' ')}\t"
                f"{cost_time:.2f}\n"
            )
            
            # 追加写入日志
            with open(self.config["LOG_FILE"], "a", encoding="utf-8") as f:
                f.write(log_line)
        
        except Exception as e:
            print(f"❌ 记录日志失败：{str(e)}")

    def extract_key_info(self, text: str) -> dict:
        """
        从结果文本中提取关键信息（适配豆包API/模型推理结果）
        :param text: 原始结果文本
        :return: 关键信息字典
        """
        try:
            key_info = {
                "关键词": [],
                "核心结论": "",
                "置信度/评分": "",
                "其他信息": ""
            }
            
            # 1. 提取置信度/评分（匹配类似"置信度：0.95"、"评分：90分"的内容）
            score_pattern = r"(置信度|评分|概率)[:：]\s*([0-9.]+)"
            score_match = re.search(score_pattern, text)
            if score_match:
                key_info["置信度/评分"] = f"{score_match.group(1)}：{score_match.group(2)}"
            
            # 2. 提取核心结论（取第一句或包含"结论"、"总结"的内容）
            conclusion_pattern = r"(结论|总结)[:：]\s*(.*?)(。|；|$)"
            conclusion_match = re.search(conclusion_pattern, text)
            if conclusion_match:
                key_info["核心结论"] = conclusion_match.group(2) + "。"
            else:
                # 无明确结论时取第一句
                sentences = re.split(r"。|！|？", text)
                key_info["核心结论"] = sentences[0] + "。" if sentences else ""
            
            # 3. 简单提取关键词（取长度>2的名词，示例逻辑，可扩展）
            word_pattern = r"([\u4e00-\u9fa5]{2,})"
            words = re.findall(word_pattern, text)
            # 去重并取前5个
            key_info["关键词"] = list(set(words))[:5]
            
            # 4. 剩余内容作为其他信息
            key_info["其他信息"] = text.replace(key_info["核心结论"], "").strip()
            
            return key_info
        
        except Exception as e:
            print(f"❌ 提取关键信息失败：{str(e)}")
            return {"关键词": [], "核心结论": text, "置信度/评分": "", "其他信息": ""}

    def _get_task_name(self, task_type: str) -> str:
        """将任务类型编码转换为中文名称"""
        task_map = {
            "chat": "智能问答",
            "classify": "文本分类",
            "sentiment": "情感分析",
            "translate": "机器翻译"
        }
        return task_map.get(task_type, task_type)

# ------------------------------
# 对外暴露的便捷函数
# ------------------------------
def format_result_simple(input_text: str, task_type: str, result_data: dict) -> str:
    """
    便捷的结果格式化函数（单例模式）
    :param input_text: 用户输入文本
    :param task_type: 任务类型
    :param result_data: 结果数据
    :return: 格式化后的文本结果
    """
    if not hasattr(format_result_simple, "processor"):
        format_result_simple.processor = ResultProcessor()
    return format_result_simple.processor.format_result(
        input_text=input_text,
        task_type=task_type,
        result_data=result_data,
        output_format="text"
    )

def save_result_simple(input_text: str, task_type: str, result_data: dict, cost_time: float) -> str:
    """
    便捷的结果保存函数
    :param input_text: 用户输入文本
    :param task_type: 任务类型
    :param result_data: 结果数据
    :param cost_time: 处理耗时
    :return: 保存路径
    """
    if not hasattr(save_result_simple, "processor"):
        save_result_simple.processor = ResultProcessor()
    return save_result_simple.processor.save_result(
        input_text=input_text,
        task_type=task_type,
        result_data=result_data,
        cost_time=cost_time,
        save_format="json"
    )

# ------------------------------
# 测试代码
# ------------------------------
if __name__ == "__main__":
    # 初始化处理器
    processor = ResultProcessor()
    
    # 测试数据
    test_input = "人工智能技术的发展给教育行业带来了巨大变革"
    test_task_type = "classify"
    test_result_data = {
        "content": "分类结果：教育（置信度：0.98）\n核心结论：人工智能对教育行业的影响显著。",
        "source": "本地NLP模型"
    }
    
    # 测试格式化
    print("===== 文本格式结果 =====")
    text_result = processor.format_result(test_input, test_task_type, test_result_data, "text")
    print(text_result)
    
    print("\n===== Markdown格式结果 =====")
    md_result = processor.format_result(test_input, test_task_type, test_result_data, "markdown")
    print(md_result)
    
    # 测试关键信息提取
    print("\n===== 关键信息提取 =====")
    key_info = processor.extract_key_info(test_result_data["content"])
    print(json.dumps(key_info, ensure_ascii=False, indent=2))
    
    # 测试保存结果
    print("\n===== 保存结果 =====")
    save_path = processor.save_result(test_input, test_task_type, test_result_data, 0.5)
    print(f"保存路径：{save_path}")
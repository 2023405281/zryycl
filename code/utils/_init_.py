# -*- coding: utf-8 -*-
"""
智能问答系统 - 工具模块包
包含文本预处理、模型辅助等通用工具
"""

# 版本标识
__version__ = "1.0.0"

# 导出核心函数/类（简化外部导入）
from .data_utils import (
    TextPreprocessor,
    preprocess_text,
    DEFAULT_CONFIG
)

# 可选：导出其他工具模块（后续扩展时添加）
# from .model_utils import load_model_weights, save_inference_result
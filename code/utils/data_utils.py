import os
import json
import jieba
import numpy as np
from collections import Counter

# 全局默认配置（与nlp_model_loader.py中的配置对齐）
DEFAULT_CONFIG = {
    "MAX_SEQ_LEN": 128,          # 文本最大长度
    "VOCAB_SIZE": 10000,         # 词汇表大小
    "PAD_TOKEN": "<PAD>",        # 填充标记
    "UNK_TOKEN": "<UNK>",        # 未知词标记
    "BOS_TOKEN": "<BOS>",        # 句子开始标记
    "EOS_TOKEN": "<EOS>",        # 句子结束标记
    "VOCAB_SAVE_PATH": "../../data/ids/vocab.json",  # 词汇表保存路径
    "ID2WORD_SAVE_PATH": "../../data/ids/id2word.json"  # ID到词映射路径
}

class TextPreprocessor:
    """文本预处理工具类，支持分词、ID映射、填充/截断、词汇表构建"""
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        # 初始化词汇表映射
        self.word2id = {}
        self.id2word = {}
        # 加载已有词汇表（若存在）
        self.load_vocab()

    def load_vocab(self):
        """加载预先生成的词汇表（word2id/id2word）"""
        try:
            # 检查词汇表文件是否存在
            if os.path.exists(self.config["VOCAB_SAVE_PATH"]) and os.path.exists(self.config["ID2WORD_SAVE_PATH"]):
                # 加载word2id
                with open(self.config["VOCAB_SAVE_PATH"], "r", encoding="utf-8") as f:
                    self.word2id = json.load(f)
                # 加载id2word
                with open(self.config["ID2WORD_SAVE_PATH"], "r", encoding="utf-8") as f:
                    self.id2word = json.load(f)
                print(f"✅ 成功加载词汇表，词汇量：{len(self.word2id)}")
            else:
                print(f"⚠️  未找到词汇表文件，将使用默认标记初始化")
                # 初始化基础标记
                self.word2id = {
                    self.config["PAD_TOKEN"]: 0,
                    self.config["UNK_TOKEN"]: 1,
                    self.config["BOS_TOKEN"]: 2,
                    self.config["EOS_TOKEN"]: 3
                }
                self.id2word = {v: k for k, v in self.word2id.items()}
        except Exception as e:
            print(f"❌ 加载词汇表失败：{str(e)}")
            # 初始化基础标记
            self.word2id = {
                self.config["PAD_TOKEN"]: 0,
                self.config["UNK_TOKEN"]: 1,
                self.config["BOS_TOKEN"]: 2,
                self.config["EOS_TOKEN"]: 3
            }
            self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, text_corpus: list, save_vocab: bool = True):
        """
        从文本语料构建词汇表
        :param text_corpus: 文本列表（如["文本1", "文本2"...]）
        :param save_vocab: 是否保存词汇表到文件
        """
        try:
            # 1. 分词并统计词频
            word_counter = Counter()
            for text in text_corpus:
                # 中文分词（jieba）
                words = jieba.lcut(text.strip())
                word_counter.update(words)
            
            # 2. 构建词汇表（保留高频词，控制词汇量）
            top_words = word_counter.most_common(self.config["VOCAB_SIZE"] - len(self.word2id))
            # 从基础标记的ID开始分配
            next_id = len(self.word2id)
            for word, _ in top_words:
                self.word2id[word] = next_id
                self.id2word[next_id] = word
                next_id += 1
            
            print(f"✅ 构建词汇表完成，总词汇量：{len(self.word2id)}")
            
            # 3. 保存词汇表到文件
            if save_vocab:
                # 创建目录（若不存在）
                os.makedirs(os.path.dirname(self.config["VOCAB_SAVE_PATH"]), exist_ok=True)
                # 保存word2id
                with open(self.config["VOCAB_SAVE_PATH"], "w", encoding="utf-8") as f:
                    json.dump(self.word2id, f, ensure_ascii=False, indent=2)
                # 保存id2word
                with open(self.config["ID2WORD_SAVE_PATH"], "w", encoding="utf-8") as f:
                    json.dump(self.id2word, f, ensure_ascii=False, indent=2)
                print(f"✅ 词汇表已保存至：{self.config['VOCAB_SAVE_PATH']}")
        
        except Exception as e:
            print(f"❌ 构建词汇表失败：{str(e)}")

    def text_to_ids(self, text: str, add_special_tokens: bool = False) -> list:
        """
        将文本转换为ID序列
        :param text: 原始文本
        :param add_special_tokens: 是否添加BOS/EOS标记
        :return: ID列表
        """
        try:
            # 1. 分词
            words = jieba.lcut(text.strip())
            # 2. 转换为ID（未知词映射为UNK）
            ids = []
            if add_special_tokens:
                ids.append(self.word2id[self.config["BOS_TOKEN"]])
            
            for word in words:
                ids.append(self.word2id.get(word, self.word2id[self.config["UNK_TOKEN"]]))
            
            if add_special_tokens:
                ids.append(self.word2id[self.config["EOS_TOKEN"]])
            
            return ids
        except Exception as e:
            print(f"❌ 文本转ID失败：{str(e)}")
            return []

    def ids_to_text(self, ids: list, remove_special_tokens: bool = True) -> str:
        """
        将ID序列转换为文本
        :param ids: ID列表
        :param remove_special_tokens: 是否移除PAD/BOS/EOS等特殊标记
        :return: 还原的文本
        """
        try:
            words = []
            special_tokens = {self.config["PAD_TOKEN"], self.config["BOS_TOKEN"], self.config["EOS_TOKEN"]}
            for idx in ids:
                word = self.id2word.get(idx, self.config["UNK_TOKEN"])
                if remove_special_tokens and word in special_tokens:
                    continue
                words.append(word)
            return "".join(words)
        except Exception as e:
            print(f"❌ ID转文本失败：{str(e)}")
            return ""

    def pad_or_truncate(self, ids: list) -> list:
        """
        对ID序列进行填充/截断，保证长度为MAX_SEQ_LEN
        :param ids: ID列表
        :return: 长度固定的ID列表
        """
        pad_id = self.word2id[self.config["PAD_TOKEN"]]
        # 截断过长序列
        if len(ids) > self.config["MAX_SEQ_LEN"]:
            ids = ids[:self.config["MAX_SEQ_LEN"]]
        # 填充过短序列
        elif len(ids) < self.config["MAX_SEQ_LEN"]:
            ids += [pad_id] * (self.config["MAX_SEQ_LEN"] - len(ids))
        return ids

    def preprocess_text(self, text: str, add_special_tokens: bool = False) -> np.ndarray:
        """
        完整的文本预处理流程（对接nlp_model_loader.py）
        :param text: 原始文本
        :param add_special_tokens: 是否添加BOS/EOS标记
        :return: 模型可输入的张量 (1, MAX_SEQ_LEN)
        """
        # 1. 文本转ID
        ids = self.text_to_ids(text, add_special_tokens)
        if not ids:
            return np.zeros((1, self.config["MAX_SEQ_LEN"]), dtype=np.int32)
        # 2. 填充/截断
        padded_ids = self.pad_or_truncate(ids)
        # 3. 转换为numpy张量（batch_size=1）
        return np.array([padded_ids], dtype=np.int32)

# ------------------------------
# 对外暴露的便捷函数（对接app.py/nlp_model_loader.py）
# ------------------------------
def preprocess_text(text: str, config: dict = None) -> np.ndarray:
    """
    便捷的文本预处理函数（单例模式）
    :param text: 原始文本
    :param config: 自定义配置（可选）
    :return: 预处理后的张量
    """
    # 单例初始化TextPreprocessor
    if not hasattr(preprocess_text, "processor"):
        preprocess_text.processor = TextPreprocessor(config)
    # 调用预处理方法
    return preprocess_text.processor.preprocess_text(text)

# ------------------------------
# 测试代码（单独运行验证功能）
# ------------------------------
if __name__ == "__main__":
    # 初始化预处理工具
    processor = TextPreprocessor()
    
    # 测试构建词汇表
    test_corpus = [
        "人工智能技术的发展给教育行业带来了巨大变革",
        "自然语言处理是人工智能的重要分支",
        "情感分析可以帮助企业了解用户需求"
    ]
    processor.build_vocab(test_corpus)
    
    # 测试文本预处理
    test_text = "人工智能改变生活"
    processed_tensor = processor.preprocess_text(test_text)
    print(f"\n原始文本：{test_text}")
    print(f"预处理后张量形状：{processed_tensor.shape}")
    print(f"预处理后张量内容：{processed_tensor[0][:10]}")  # 打印前10个ID
    
    # 测试ID转文本
    test_ids = processed_tensor[0].tolist()
    restored_text = processor.ids_to_text(test_ids)
    print(f"还原文本：{restored_text}")
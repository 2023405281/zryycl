# -*- coding: utf-8 -*-
"""
智能问答系统 - Seq2Seq模型实现
包含编码器(Encoder)、解码器(Decoder)、注意力机制(可选)，支持训练/推理
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import numpy as np
import os
import json

# 模型配置（与data_utils/模型加载模块对齐）
SEQ2SEQ_CONFIG = {
    "EMBEDDING_DIM": 256,        # 嵌入层维度
    "UNITS": 1024,               # LSTM单元数
    "MAX_SEQ_LEN": 128,          # 最大序列长度
    "VOCAB_SIZE": 10000,         # 词汇表大小
    "BATCH_SIZE": 64,            # 批次大小
    "EPOCHS": 20,                # 训练轮数
    "CHECKPOINT_DIR": "../model/seq2seq_checkpoint",  # 检查点保存目录
    "PAD_TOKEN_ID": 0,           # PAD标记ID
    "UNK_TOKEN_ID": 1,           # UNK标记ID
    "BOS_TOKEN_ID": 2,           # BOS标记ID
    "EOS_TOKEN_ID": 3            # EOS标记ID
}

class Encoder(Model):
    """Seq2Seq编码器（LSTM实现）"""
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # 嵌入层
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        # LSTM层（返回输出和状态）
        self.lstm = layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        """
        前向传播
        :param x: 输入序列 (batch_size, seq_len)
        :param hidden: 初始隐藏状态 (batch_size, enc_units)
        :return: output, state_h, state_c
        """
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, state_h, state_c

    def initialize_hidden_state(self):
        """初始化隐藏状态"""
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class BahdanauAttention(layers.Layer):
    """Bahdanau注意力机制（可选）"""
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        """
        计算注意力权重
        :param query: 解码器隐藏状态 (batch_size, units)
        :param values: 编码器输出 (batch_size, seq_len, units)
        :return: 上下文向量, 注意力权重
        """
        # query扩展维度: (batch_size, 1, units)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # 注意力分数: (batch_size, seq_len, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)
        ))
        
        # 注意力权重: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # 上下文向量: (batch_size, units)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class Decoder(Model):
    """Seq2Seq解码器（带注意力机制）"""
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # 嵌入层
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        # LSTM层
        self.lstm = layers.LSTM(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        # 全连接层（输出词汇表维度）
        self.fc = layers.Dense(vocab_size)
        # 注意力机制
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        """
        前向传播
        :param x: 解码器输入 (batch_size, 1)
        :param hidden: 解码器隐藏状态 (batch_size, dec_units)
        :param enc_output: 编码器输出 (batch_size, seq_len, enc_units)
        :return: 预测输出, 状态h, 状态c, 注意力权重
        """
        # 计算注意力上下文向量
        context_vector, attention_weights = self.attention(hidden[0], enc_output)
        
        # 嵌入层处理
        x = self.embedding(x)
        
        # 拼接上下文向量和嵌入输出: (batch_size, 1, embedding_dim + dec_units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # LSTM前向传播
        output, state_h, state_c = self.lstm(x)
        
        # 调整形状: (batch_size * 1, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # 全连接层预测: (batch_size, vocab_size)
        x = self.fc(output)
        
        return x, [state_h, state_c], attention_weights

class Seq2SeqModel:
    """Seq2Seq模型封装类：训练/推理/保存/加载"""
    def __init__(self, config: dict = None):
        self.config = config or SEQ2SEQ_CONFIG
        # 初始化编码器/解码器
        self.encoder = Encoder(
            self.config["VOCAB_SIZE"],
            self.config["EMBEDDING_DIM"],
            self.config["UNITS"],
            self.config["BATCH_SIZE"]
        )
        self.decoder = Decoder(
            self.config["VOCAB_SIZE"],
            self.config["EMBEDDING_DIM"],
            self.config["UNITS"],
            self.config["BATCH_SIZE"]
        )
        # 优化器和损失函数
        self.optimizer = optimizers.Adam()
        self.loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        # 检查点管理器
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.config["CHECKPOINT_DIR"],
            max_to_keep=5
        )

    def loss_function(self, real, pred):
        """自定义损失函数（忽略PAD标记）"""
        mask = tf.math.logical_not(tf.math.equal(real, self.config["PAD_TOKEN_ID"]))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        """单步训练（装饰为TF图函数加速）"""
        loss = 0
        
        with tf.GradientTape() as tape:
            # 编码器前向传播
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)
            dec_hidden = [enc_h, enc_c]
            
            # 解码器输入初始化为BOS标记
            dec_input = tf.expand_dims([self.config["BOS_TOKEN_ID"]] * self.config["BATCH_SIZE"], 1)
            
            # 教师强制训练（Teacher Forcing）
            for t in range(1, targ.shape[1]):
                # 解码器前向传播
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                # 计算损失
                loss += self.loss_function(targ[:, t], predictions)
                # 教师强制：使用真实目标作为下一个输入
                dec_input = tf.expand_dims(targ[:, t], 1)
        
        # 计算平均损失
        batch_loss = (loss / int(targ.shape[1]))
        # 获取可训练变量
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        # 计算梯度并优化
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss

    def train(self, dataset, epochs: int = None):
        """
        训练模型
        :param dataset: tf.data.Dataset格式的训练数据 (inp, targ)
        :param epochs: 训练轮数（默认使用配置中的EPOCHS）
        """
        epochs = epochs or self.config["EPOCHS"]
        # 加载最新检查点（若存在）
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"✅ 恢复最新检查点：{self.checkpoint_manager.latest_checkpoint}")
        
        # 开始训练
        for epoch in range(epochs):
            start = tf.timestamp()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            
            # 遍历批次
            for (batch, (inp, targ)) in enumerate(dataset):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                
                if batch % 100 == 0:
                    print(f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}")
            
            # 保存检查点
            if (epoch + 1) % 2 == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print(f"✅ 保存检查点至：{ckpt_save_path}")
            
            print(f"Epoch {epoch+1} Loss {total_loss/len(dataset):.4f}")
            print(f"耗时：{tf.timestamp() - start:.2f}秒\n")

    def infer(self, input_ids: list, word2id: dict, id2word: dict, max_len: int = None) -> str:
        """
        推理（预测）函数
        :param input_ids: 输入文本的ID序列
        :param word2id: 词汇表（word->id）
        :param id2word: 词汇表（id->word）
        :param max_len: 最大生成长度（默认使用MAX_SEQ_LEN）
        :return: 生成的文本结果
        """
        max_len = max_len or self.config["MAX_SEQ_LEN"]
        
        # 调整输入形状适配批量大小=1
        input_tensor = tf.convert_to_tensor([input_ids])
        enc_hidden = [tf.zeros((1, self.config["UNITS"])), tf.zeros((1, self.config["UNITS"]))]
        enc_output, enc_h, enc_c = self.encoder(input_tensor, enc_hidden)
        dec_hidden = [enc_h, enc_c]
        
        # 解码器初始输入（BOS标记）
        dec_input = tf.expand_dims([word2id["<BOS>"]], 0)
        result = []
        attention_weights = []
        
        # 逐步生成
        for t in range(max_len):
            predictions, dec_hidden, attn_weights = self.decoder(dec_input, dec_hidden, enc_output)
            
            # 记录注意力权重
            attention_weights.append(attn_weights)
            
            # 取概率最大的ID
            predicted_id = tf.argmax(predictions[0]).numpy()
            
            # 遇到EOS标记停止生成
            if predicted_id == word2id["<EOS>"]:
                break
            
            # 转换为词并添加到结果
            result.append(id2word.get(predicted_id, "<UNK>"))
            
            # 作为下一个输入
            dec_input = tf.expand_dims([predicted_id], 0)
        
        # 拼接结果
        return "".join(result)

    def save_model_as_h5(self, save_path: str):
        """
        保存模型为.h5格式（适配nlp_model_loader.py加载）
        :param save_path: 保存路径（如../model/seq2seq_model.h5）
        """
        try:
            # 保存编码器
            encoder_save_path = save_path.replace(".h5", "_encoder.h5")
            self.encoder.save(encoder_save_path, save_format="h5")
            # 保存解码器
            decoder_save_path = save_path.replace(".h5", "_decoder.h5")
            self.decoder.save(decoder_save_path, save_format="h5")
            print(f"✅ Seq2Seq模型已保存：")
            print(f"   - 编码器：{encoder_save_path}")
            print(f"   - 解码器：{decoder_save_path}")
            
            # 保存配置
            config_save_path = save_path.replace(".h5", "_config.json")
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"   - 配置文件：{config_save_path}")
        
        except Exception as e:
            print(f"❌ 保存.h5模型失败：{str(e)}")

    def load_h5_model(self, encoder_path: str, decoder_path: str):
        """
        从.h5文件加载模型
        :param encoder_path: 编码器.h5路径
        :param decoder_path: 解码器.h5路径
        """
        try:
            self.encoder = load_model(encoder_path, compile=False)
            self.decoder = load_model(decoder_path, compile=False)
            print(f"✅ 成功加载Seq2Seq模型：")
            print(f"   - 编码器：{encoder_path}")
            print(f"   - 解码器：{decoder_path}")
        except Exception as e:
            print(f"❌ 加载.h5模型失败：{str(e)}")

# ------------------------------
# 便捷函数（对接nlp_model_loader.py）
# ------------------------------
def load_seq2seq_model(model_dir: str = None) -> Seq2SeqModel:
    """
    加载Seq2Seq模型（从检查点或.h5文件）
    :param model_dir: 模型目录（默认使用配置中的CHECKPOINT_DIR）
    :return: Seq2SeqModel实例
    """
    model = Seq2SeqModel()
    model_dir = model_dir or SEQ2SEQ_CONFIG["CHECKPOINT_DIR"]
    
    # 优先加载检查点
    if os.path.exists(model_dir):
        ckpt_manager = tf.train.CheckpointManager(
            model.checkpoint, model_dir, max_to_keep=5
        )
        if ckpt_manager.latest_checkpoint:
            model.checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"✅ 从检查点加载Seq2Seq模型成功")
            return model
    
    # 尝试加载.h5文件
    encoder_h5 = os.path.join(model_dir, "encoder.h5")
    decoder_h5 = os.path.join(model_dir, "decoder.h5")
    if os.path.exists(encoder_h5) and os.path.exists(decoder_h5):
        model.load_h5_model(encoder_h5, decoder_h5)
        return model
    
    print(f"⚠️  未找到Seq2Seq模型文件，返回初始化模型")
    return model

def seq2seq_infer(text: str, text_preprocessor: object, model: Seq2SeqModel) -> str:
    """
    Seq2Seq推理便捷函数
    :param text: 输入文本
    :param text_preprocessor: TextPreprocessor实例（来自data_utils.py）
    :param model: Seq2SeqModel实例
    :return: 生成的回复文本
    """
    # 文本预处理
    input_ids = text_preprocessor.text_to_ids(text, add_special_tokens=True)
    # 填充到固定长度
    input_ids = text_preprocessor.pad_or_truncate(input_ids)
    # 推理
    result = model.infer(
        input_ids,
        text_preprocessor.word2id,
        text_preprocessor.id2word
    )
    return result

# ------------------------------
# 测试代码
# ------------------------------
if __name__ == "__main__":
    # 初始化模型
    seq2seq_model = Seq2SeqModel()
    print("✅ Seq2Seq模型初始化完成")
    
    # 测试推理流程（需先训练或加载模型）
    from utils.data_utils import TextPreprocessor
    processor = TextPreprocessor()
    # 构建测试词汇表
    test_corpus = ["你好", "人工智能", "智能问答系统"]
    processor.build_vocab(test_corpus)
    
    # 测试推理（仅演示流程，未训练模型会输出随机结果）
    test_text = "你好"
    result = seq2seq_infer(test_text, processor, seq2seq_model)
    print(f"\n输入：{test_text}")
    print(f"输出：{result}")
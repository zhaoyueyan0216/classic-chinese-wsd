import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

# 服务器上的模型本地路径（优先从环境变量读取）
DEFAULT_MODEL_NAME = os.environ.get(
    "GUWENBERT_MODEL_PATH",
    "ethanyt/guwenbert-base"  # set GUWENBERT_MODEL_PATH to a local snapshot on HPC
)

# 古汉语BERT编码器
class GuWenBERTEncoder:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=64):
        print(f"   📂 从本地snapshot加载模型...")
        print(f"   🔍 模型路径: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=False,  # 强制使用 .bin 文件，避免 safetensors/bin 检查
            dtype=torch.float16,  # 使用 fp16 减少显存占用
        )
        print(f"   ✅ 本地模型加载成功")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🧠 使用设备: {self.device}")
        print(f"   📦 批处理大小: {batch_size}")
        if torch.cuda.is_available():
            print(f"   🚀 GPU加速: 可用 ({torch.cuda.get_device_name(0)})")
            print(f"   📊 GPU内存: {torch.cuda.memory_allocated()/1024/1024:.2f} MB 已分配")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """使用批处理方式对文本进行编码，提高处理速度"""
        embeddings = []
        
        # 将文本分批次处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # 对批次文本进行tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,  # 批次内填充到相同长度
            )
            
            # 将输入移到设备上
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播获取输出
            outputs = self.model(**inputs)
            
            # 提取CLS向量（句首标记的向量表示）
            cls_vecs = outputs.last_hidden_state[:, 0, :]
            
            # 将结果移回CPU并转换为numpy数组
            embeddings.append(cls_vecs.cpu().numpy())
        
        # 合并所有批次的结果
        return np.vstack(embeddings)

# 初始化全局编码器实例
guwen_encoder = None


def get_encoder():
    """延迟初始化编码器"""
    global guwen_encoder
    if guwen_encoder is None:
        print("\n🔧 初始化古汉语BERT编码器...")
        guwen_encoder = GuWenBERTEncoder()
        print("✅ 古汉语BERT编码器初始化完成！")
    return guwen_encoder


def get_embedding(text, max_length=128):
    """
    获取文本的古汉语BERT嵌入向量
    
    Args:
        text (str): 输入文本
        max_length (int): 最大序列长度
        
    Returns:
        np.ndarray: 文本的嵌入向量，如果处理失败则返回全零向量
    """
    try:
        encoder = get_encoder()
        # 使用古汉语BERT编码器获取嵌入
        embedding = encoder.encode([text])[0]
        
        # 归一化向量
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    except Exception as e:
        print(f"[Embedding计算] 处理句子时出错: {text[:20]}... | 错误: {str(e)}")
        # 返回全零向量作为fallback
        return np.zeros(768)


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    Args:
        vec1 (np.ndarray): 第一个向量
        vec2 (np.ndarray): 第二个向量

    Returns:
        float: 余弦相似度值
    """
    if vec1 is None or vec2 is None:
        return 0.0

    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    if hasattr(similarity, 'item'):
        return float(similarity.item())
    return float(similarity)


def get_embeddings(texts, batch_size=128):
    """
    批量获取文本嵌入，真正提高GPU利用率

    Args:
        texts (list): 输入文本列表
        batch_size (int): 批处理大小，默认128，如果爆显存可改为64

    Returns:
        np.ndarray: 文本的嵌入向量矩阵 (len(texts), 768)
    """
    if not texts:
        return np.zeros((0, 768))

    try:
        encoder = get_encoder()
        encoder.batch_size = batch_size

        embeddings = encoder.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        return embeddings
    except Exception as e:
        print(f"[Embedding批量计算] 出错: {e}")
        return np.zeros((len(texts), 768))
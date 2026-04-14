import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

# 设置 Hugging Face 镜像（如果环境变量未设置）
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 固定的模型名称
DEFAULT_MODEL_NAME = "ethanyt/guwenbert-base"

# 古汉语BERT编码器
class GuWenBERTEncoder:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=64):
        # 确保使用正确的模型名称格式
        if os.path.isabs(model_name) or model_name.startswith('./') or model_name.startswith('../'):
            print(f"   ⚠️ 检测到本地路径，使用默认模型: {DEFAULT_MODEL_NAME}")
            model_name = DEFAULT_MODEL_NAME
        
        try:
            # 首先尝试从本地加载
            print(f"   📂 尝试从本地加载模型...")
            print(f"   🔍 模型名称: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            print(f"   ✅ 成功从本地加载模型")
        except Exception as e:
            print(f"   ⚠️ 本地模型未找到，尝试从镜像下载...")
            print(f"   🌐 使用镜像: {os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=False)
                print(f"   ✅ 成功从镜像下载模型")
            except Exception as e2:
                print(f"   ❌ 无法加载模型: {e2}")
                print(f"   💡 请手动下载模型并放到缓存目录")
                print(f"   📁 缓存目录: ~/.cache/huggingface/hub/")
                print(f"   🔗 模型地址: https://hf-mirror.com/ethanyt/guwenbert-base")
                raise e2
        
        # 将模型移到可用的设备上（GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🧠 使用设备: {self.device}")
        print(f"   📦 批处理大小: {batch_size}")
        if torch.cuda.is_available():
            print(f"   🚀 GPU加速: 可用 ({torch.cuda.get_device_name(0)})")
            print(f"   📊 GPU内存: {torch.cuda.memory_allocated()/1024/1024:.2f} MB 已分配")
        self.model.to(self.device)
        self.model.eval()
        # 设置批处理大小
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
    
    # 确保向量是一维的
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算向量的范数
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 防止除以零
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    
    # 确保返回标量值 - 使用.item()获取numpy标量的Python内置类型，然后转换为float
    if hasattr(similarity, 'item'):
        return float(similarity.item())
    return float(similarity)

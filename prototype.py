import numpy as np

def build_prototype(sent_embs):
    """
    从多个句子的嵌入向量构建prototype向量
    
    Args:
        sent_embs (list): 句子嵌入向量的列表
        
    Returns:
        np.ndarray: prototype向量
    """
    if not sent_embs or len(sent_embs) == 0:
        return None
    
    # 将嵌入向量转换为numpy数组
    embs_array = np.array(sent_embs)
    
    # 确保embs_array是二维的（样本数 x 向量维度）
    if embs_array.ndim == 1:
        embs_array = embs_array.reshape(1, -1)
    
    # 计算平均向量作为prototype
    prototype = np.mean(embs_array, axis=0)
    
    # 归一化prototype向量
    norm = np.linalg.norm(prototype)
    if norm > 0:
        prototype = prototype / norm
    
    # 确保prototype是一维数组
    prototype = prototype.flatten()
    
    return prototype
import re

def extract_grammar_role(sentence, target_word):
    """
    提取目标词在句子中的语法角色
    
    Args:
        sentence (str): 包含目标词的句子
        target_word (str): 目标词
        
    Returns:
        str: 语法角色描述
    """
    # 古汉语语法规则简化版
    patterns = [
        # 主语模式：[主语] + 谓语
        (r'^[^，。！？]*{}[^，。！？]*[，。！？]'.format(target_word), '主语'),
        # 谓语模式：主语 + [谓语]
        (r'[，。！？][^，。！？]*{}[^，。！？]*[，。！？]'.format(target_word), '谓语'),
        # 宾语模式：谓语 + [宾语]
        (r'[^，。！？]*[之其矣乎也][^，。！？]*{}[^，。！？]*[，。！？]'.format(target_word), '宾语'),
        # 定语模式：[定语] + 名词
        (r'[^，。！？]*{}[之的][^，。！？]*[，。！？]'.format(target_word), '定语'),
        # 状语模式：[状语] + 动词
        (r'[^，。！？]*{}[乎矣也][^，。！？]*[，。！？]'.format(target_word), '状语'),
    ]
    
    for pattern, role in patterns:
        if re.search(pattern, sentence):
            return role
    
    return '未知'

def extract_collocations(sentence, target_word, window_size=2):
    """
    提取目标词的搭配词
    
    Args:
        sentence (str): 包含目标词的句子
        target_word (str): 目标词
        window_size (int): 窗口大小
        
    Returns:
        list: 搭配词列表
    """
    if target_word not in sentence:
        return []
    
    # 获取目标词在句子中的位置
    target_pos = sentence.index(target_word)
    
    # 提取目标词前后的窗口
    start_pos = max(0, target_pos - window_size)
    end_pos = min(len(sentence), target_pos + len(target_word) + window_size)
    
    # 提取窗口内的所有字符（简化版，不做分词）
    window = sentence[start_pos:end_pos]
    
    # 排除目标词本身
    collocations = [char for char in window if char != target_word]
    
    return collocations

def build_usage_schema(sentences, target_word=None, max_collocations=5):
    """
    从多个句子中构建usage schema
    
    Args:
        sentences (list): 包含目标词的句子列表
        target_word (str): 目标词
        max_collocations (int): 最大搭配词数量
        
    Returns:
        dict: usage schema字典，包含语法角色、常见搭配和典型情境
    """
    if not sentences or len(sentences) == 0:
        return {
            'grammar_roles': {},
            'common_collocations': [],
            'typical_contexts': []
        }
    
    # 提取语法角色
    grammar_roles = {}
    if target_word:
        for sentence in sentences:
            role = extract_grammar_role(sentence, target_word)
            grammar_roles[role] = grammar_roles.get(role, 0) + 1
    
    # 提取搭配词
    collocations = {}
    if target_word:
        for sentence in sentences:
            words = extract_collocations(sentence, target_word)
            for word in words:
                collocations[word] = collocations.get(word, 0) + 1
    
    # 按频率排序，获取常见搭配
    sorted_collocations = sorted(collocations.items(), key=lambda x: x[1], reverse=True)
    top_collocations = [word for word, count in sorted_collocations[:max_collocations]]
    
    # 提取典型情境（取前3个最具代表性的句子）
    typical_contexts = sentences[:3]
    
    # 生成简版的usage schema
    schema = {
        'grammar_roles': grammar_roles,
        'common_collocations': top_collocations,
        'typical_contexts': typical_contexts
    }
    
    return schema
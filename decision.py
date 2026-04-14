from embedding_utils import get_embedding, cosine_similarity
import numpy as np

def decide_sense(context_text, sense_A, sense_B):
    """
    比较两个义项与当前上下文的匹配程度，做出最终裁决
    重点：将历史证据原文和schema都给大模型，让大模型二次判断
    
    Args:
        context_text (str): 当前上下文文本
        sense_A (dict): 第一个义项的证据，包含prototype、schema和sim
        sense_B (dict): 第二个义项的证据，包含prototype、schema和sim
        
    Returns:
        dict: 最终裁决结果，包含预测义项、是否排除了另一个以及理由
    """
    from llm import call_llm as llm_inference
    
    # 步骤1：提取两个义项的详细信息
    def extract_sense_info(sense, index):
        info = f"义项{index} (wsid={sense['wsid']}):\n"
        
        # 提取历史证据原文
        if sense.get('prototype'):
            prototype = sense['prototype']
            if isinstance(prototype, dict):
                if 'definition' in prototype:
                    info += f"  定义: {prototype['definition']}\n"
                if 'explanation' in prototype:
                    info += f"  解释: {prototype['explanation']}\n"
                if 'examples' in prototype:
                    examples = prototype['examples']
                    if examples:
                        info += "  历史例句:\n"
                        for i, example in enumerate(examples[:3]):  # 最多3个例子
                            if isinstance(example, dict):
                                if 'text' in example:
                                    info += f"    {i+1}. {example['text']}\n"
                                elif 'context' in example:
                                    info += f"    {i+1}. {example['context']}\n"
                                elif 'sentence' in example:
                                    info += f"    {i+1}. {example['sentence']}\n"
                            elif isinstance(example, str):
                                info += f"    {i+1}. {example}\n"
            elif isinstance(prototype, (list, np.ndarray)):
                # 旧格式的prototype向量，添加默认信息
                info += "  定义: 无详细定义\n"
                info += "  解释: 基于向量表示的原型\n"
                info += "  历史例句: 无详细例句\n"
        
        # 提取schema信息
        if sense.get('schema'):
            schema = sense['schema']
            info += "  Schema信息:\n"
            if isinstance(schema, dict):
                if 'grammar_roles' in schema:
                    info += f"    语法角色: {schema['grammar_roles']}\n"
                if 'common_collocations' in schema:
                    info += f"    常见搭配: {schema['common_collocations']}\n"
                if 'typical_contexts' in schema:
                    info += f"    典型情境: {schema['typical_contexts']}\n"
            else:
                # 如果schema不是字典格式，添加默认信息
                info += f"    {schema}\n"
        else:
            # 如果没有schema信息，添加默认信息
            info += "  Schema信息: 无\n"
        
        return info
    
    # 步骤2：构建大模型二次判断的提示词
    sense_A_info = extract_sense_info(sense_A, "A")
    sense_B_info = extract_sense_info(sense_B, "B")
    
    prompt = f"""你是一位古汉语词义消歧专家，现在需要对两个候选义项进行最终判断。

当前上下文：
{context_text}

请仔细分析以下两个义项的详细信息，结合上下文内容，判断哪个义项更适合当前语境：

{sense_A_info}

{sense_B_info}

判断要求：
1. 结合历史证据原文（定义、解释、例句）和Schema信息进行综合分析
2. 重点考虑历史证据与当前上下文的语义关联性
3. 考虑Schema信息是否符合当前上下文的语法和语义环境
4. 给出明确的判断结果（A或B）和详细理由

请按照以下格式输出：
判断结果：[A/B]
详细理由：[你的分析理由]
"""
    
    # 步骤3：调用大模型进行二次判断
    try:
        response = llm_inference(prompt)
        
        # 步骤4：解析大模型的判断结果
        if "判断结果：A" in response or "判断结果:A" in response:
            winner = sense_A
            loser = sense_B
            reason = "大模型二次判断选择了义项A"
        elif "判断结果：B" in response or "判断结果:B" in response:
            winner = sense_B
            loser = sense_A
            reason = "大模型二次判断选择了义项B"
        else:
            # 解析失败，默认选择第一个
            winner = sense_A
            loser = sense_B
            reason = "大模型判断结果解析失败，默认选择第一个义项"
            
    except Exception as e:
        # 大模型调用失败，默认选择第一个
        winner = sense_A
        loser = sense_B
        reason = f"大模型调用失败: {str(e)}，默认选择第一个义项"
        response = f"大模型调用失败: {str(e)}"
    
    # 步骤5：生成详细理由（包含大模型的分析）
    detailed_reason = f"当前上下文：{context_text}\n"
    detailed_reason += f"\n[判断依据] 大模型二次判断，结合历史证据原文和Schema信息\n"
    detailed_reason += f"\n义项A (wsid={sense_A['wsid']})：\n{sense_A_info}\n"
    detailed_reason += f"义项B (wsid={sense_B['wsid']})：\n{sense_B_info}\n"
    detailed_reason += f"\n大模型判断结果：\n{response}\n"
    detailed_reason += f"\n最终裁决：{reason}\n"
    
    # 步骤6：确定是否排除另一个义项
    exclude = "排除" in response or "排除" in reason
    
    # 返回最终裁决结果
    return {
        "prediction": winner['wsid'],
        "exclude_other": exclude,
        "reason": reason,
        "detailed_reason": detailed_reason,
        "llm_response": response
    }

def evaluate_schema_match(schema, context_text):
    """
    评估usage schema与当前上下文的匹配程度
    
    Args:
        schema (dict): 义项的usage schema
        context_text (str): 当前上下文文本
        
    Returns:
        float: schema匹配度得分（0-1之间）
    """
    if not schema:
        return 0.0
    
    score = 0.0
    match_count = 0
    total_metrics = 3  # 语法角色、常见搭配、典型情境
    
    # 1. 检查语法角色匹配
    if 'grammar_roles' in schema and schema['grammar_roles']:
        # 简化版：只要schema中存在语法角色信息，就给一定分数
        score += 0.33
        match_count += 1
    
    # 2. 检查常见搭配匹配
    if 'common_collocations' in schema and schema['common_collocations']:
        collocations = schema['common_collocations']
        matched_collocations = [word for word in collocations if word in context_text]
        if matched_collocations:
            # 匹配的搭配词越多，分数越高
            collocation_score = len(matched_collocations) / len(collocations) * 0.33
            score += collocation_score
            match_count += 1
    
    # 3. 检查典型情境匹配
    if 'typical_contexts' in schema and schema['typical_contexts']:
        # 简化版：只要schema中存在典型情境信息，就给一定分数
        score += 0.33
        match_count += 1
    
    # 如果没有任何匹配，返回0
    if match_count == 0:
        return 0.0
    
    # 确保返回的是标量float
    return float(score)
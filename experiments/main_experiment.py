# Comparison test for full vs ablation_no_history configurations with paired t-test (Local version)
# Key improvement: Top-k union is generated once per sample, then shared by both configurations
# Updated: Replaced schema with historical usage summary
# Added: candidate_count, pairwise similarity metrics, case_type classification

import json
import sys
import io
import random
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

# Set default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

BASE_DIR = Path(__file__).parent.parent
from collections import defaultdict
from core.embedding_utils import get_embedding, cosine_similarity
from core.pseudo_retrieval import get_supporting_sentences_with_pseudo_bm25, get_supporting_sentences_without_pseudo_bm25
from core.llm import call_llm, cached_call_llm


# Base configuration
BASE_CONFIG = {
    "use_top2_filter": True,
    "use_pseudo_query": True,
    "use_bm25": True,
    "use_rerank": True,
    "use_schema": False,  # Updated: No longer using schema
    "use_similarity": True,
    "use_supporting_sentences": True,
    "final_decision_with_llm": True,
    "use_final_cache": False
}

# Different experiment configurations
CONFIGS = {
    # Full configuration
    "full": BASE_CONFIG.copy(),

    # Ablation 3: No historical evidence
    "ablation_no_history": BASE_CONFIG.copy(),

    # Pure LLM: Direct selection without Top-k union or historical evidence
    "pure_llm": BASE_CONFIG.copy(),

    # Ablation A: With aggregator, but no sense agent verification
    "ablation_no_verify": BASE_CONFIG.copy(),

    # Ablation B: With sense agent verification, but no aggregator (use voting instead)
    "ablation_no_aggregator": BASE_CONFIG.copy(),
}

# Set ablation configurations
CONFIGS["ablation_no_history"]["use_supporting_sentences"] = False
CONFIGS["ablation_no_history"]["use_schema"] = False
CONFIGS["ablation_no_history"]["use_similarity"] = False
CONFIGS["ablation_no_history"]["use_final_cache"] = False

# Set pure LLM configuration (no Top-k union, no history evidence)
CONFIGS["pure_llm"]["use_top2_filter"] = False  # Don't use Top-k union
CONFIGS["pure_llm"]["use_supporting_sentences"] = False
CONFIGS["pure_llm"]["use_schema"] = False
CONFIGS["pure_llm"]["use_similarity"] = False
CONFIGS["pure_llm"]["use_final_cache"] = False

# Ablation A: No agent verification (but still use aggregator)
CONFIGS["ablation_no_verify"]["use_agent_verify"] = False

# Ablation B: No aggregator (but still use agent verification with voting)
CONFIGS["ablation_no_aggregator"]["use_aggregator"] = False


def load_test_data():
    """
    Load test data

    Returns:
        tuple: (contexts, senses, wsid_to_context, wsid_to_sense, word_to_senses, preloaded_corpus)
    """
    # Load test contexts from local path
    contexts_path = BASE_DIR / "dataprocess" / "data" / "contexts.json"
    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts_data = json.load(f)
        # Extract RECORDS array
        contexts = contexts_data.get("RECORDS", [])

    # Load test senses
    with open(BASE_DIR / "data" / "character_senses.json", "r", encoding="utf-8") as f:
        senses = json.load(f)["RECORDS"]

    # Build mapping from wsid to context
    wsid_to_context = {}
    for ctx in contexts:
        wsid = str(ctx["wsid"])
        wsid_to_context[wsid] = ctx

    # Build mapping from wsid to sense
    wsid_to_sense = {}
    for sense in senses:
        wsid = str(sense["wsid"])
        wsid_to_sense[wsid] = sense

    # Build mapping from word to senses
    word_to_senses = defaultdict(list)
    for sense in senses:
        word = sense["word"]
        word_to_senses[word].append(sense)

    # Preloaded corpus is not needed due to Pyserini BM25 retrieval
    preloaded_corpus = []

    return contexts, senses, wsid_to_context, wsid_to_sense, word_to_senses, preloaded_corpus


def calculate_sense_similarity_metrics(candidate_senses):
    """
    Calculate pairwise similarity metrics for candidate senses
    
    Args:
        candidate_senses (list): List of candidate senses
        
    Returns:
        dict: Dictionary containing candidate_count, mean_pairwise_similarity, max_pairwise_similarity, top2_similarity
    """
    candidate_count = len(candidate_senses)
    
    if candidate_count < 2:
        return {
            'candidate_count': candidate_count,
            'mean_pairwise_similarity': 0.0,
            'max_pairwise_similarity': 0.0,
            'top2_similarity': 0.0
        }
    
    # Get embeddings for all sense glosses
    glosses = [s.get("newgloss", "") for s in candidate_senses]
    embeddings = [get_embedding(gloss) for gloss in glosses]
    
    # Calculate all pairwise similarities
    similarities = []
    for i in range(candidate_count):
        for j in range(i + 1, candidate_count):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    # Sort similarities in descending order
    similarities.sort(reverse=True)
    
    mean_pairwise_similarity = np.mean(similarities) if similarities else 0.0
    max_pairwise_similarity = similarities[0] if similarities else 0.0
    top2_similarity = similarities[1] if len(similarities) >= 2 else 0.0
    
    return {
        'candidate_count': candidate_count,
        'mean_pairwise_similarity': mean_pairwise_similarity,
        'max_pairwise_similarity': max_pairwise_similarity,
        'top2_similarity': top2_similarity
    }


def determine_case_type(full_correct, no_history_correct):
    """
    Determine case type based on full and no_history correctness
    
    Args:
        full_correct (bool): Whether full configuration is correct
        no_history_correct (bool): Whether no_history configuration is correct
        
    Returns:
        str: Case type (A, B, C, or D)
    """
    if full_correct and not no_history_correct:
        return 'A'  # full correct, no_history wrong
    elif not full_correct and no_history_correct:
        return 'B'  # no_history correct, full wrong
    elif full_correct and no_history_correct:
        return 'C'  # both correct
    else:
        return 'D'  # both wrong


def analyze_history_quality_v2(agent_details_str):
    """
    Analyze the quality of historical evidence based on agent_details

    Args:
        agent_details_str (str): JSON string of agent details

    Returns:
        str: History quality label
    """
    try:
        agents = json.loads(agent_details_str)
        if not agents:
            return "no_evidence"

        total_senses = len(agents)
        empty_count = sum(1 for a in agents if a.get("valid_count", 0) == 0)
        empty_ratio = empty_count / total_senses

        if empty_ratio == 0:
            return "full_evidence"
        elif empty_ratio < 0.5:
            return "partial_evidence"
        elif empty_ratio < 1.0:
            return "poor_evidence"
        else:
            return "no_evidence"
    except:
        return "no_evidence"


def sense_agent(text, target_word, sense_gloss, sense_wsid,
                 raw_sentences, cached_call_llm):
    """
    义项智能体：验证证据并判断目标句是否符合该义项
    对应MADAM-RAG里每个文档对应一个独立智能体的设计
    """
    # 基础过滤：只保留含目标词的句子
    candidates = [
        s for s in raw_sentences
        if target_word in (s["sentence"] if isinstance(s, dict) else str(s))
    ]

    # 没有任何含目标词的句子
    if not candidates:
        return {
            "wsid": sense_wsid,
            "gloss": sense_gloss,
            "valid_examples": [],
            "evidence_status": "unknown",
            "contextual_fit": "unknown",
            "support": None,
            "confidence": "unknown",
            "reason": "未检索到含目标词的历史例句；这表示证据缺失，不表示该义项不符合目标句。"
        }

    # 构建编号例句
    numbered = "\n".join(
        f"{i+1}. {s['sentence']}"
        for i, s in enumerate(candidates[:10])
    )

    prompt = f"""你是古汉语词义专家。

目标句：{text}
目标词：「{target_word}」
候选义项：{sense_gloss}

以下是为该义项检索到的历史例句：
{numbered}

请完成两个任务：

【任务1】筛选有效例句
从上面的例句中，找出「{target_word}」确实体现了「{sense_gloss}」这个义项的句子。
注意：
- 语境高度相似的例句只选1句代表（避免重复）
- 无法判断义项归属的句子不选
- 宁可一句不选，也不选不确定的

【任务2】判断目标句
参考筛选出的有效例句的语境，判断目标句中「{target_word}」是否符合「{sense_gloss}」这个义项。
如果有效例句为空，仅根据义项定义和目标句语境判断。

输出严格JSON，不要输出其他内容：
{{
  "valid_examples": ["例句原文", ...],
  "evidence_status": "supported或noisy或unknown",
  "contextual_fit": "high或medium或low",
  "support": true或false或null,
  "confidence": "high或medium或low或unknown",
  "reason": "一句话说明：先判断目标句与义项定义是否契合，再说明历史例句是否提供辅助支持"
}}"""

    cache_tag = f"agent|{sense_wsid}|{hash(text[:50]) % 100000}"
    response = cached_call_llm(prompt, cache_tag=cache_tag, seed=42)

    # 清理markdown包装
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        result = json.loads(response)
        return {
            "wsid": sense_wsid,
            "gloss": sense_gloss,
            "valid_examples": result.get("valid_examples", []),
            "evidence_status": result.get("evidence_status", "unknown"),
            "contextual_fit": result.get("contextual_fit", "unknown"),
            "support": result.get("support", None),
            "confidence": result.get("confidence", "unknown"),
            "reason": result.get("reason", "")
        }
    except Exception as e:
        return {
            "wsid": sense_wsid,
            "gloss": sense_gloss,
            "valid_examples": [],
            "evidence_status": "unknown",
            "contextual_fit": "unknown",
            "support": None,
            "confidence": "unknown",
            "reason": f"解析失败: {e}"
        }


def sense_agent_no_verify(text, target_word, sense_gloss,
                          sense_wsid, raw_sentences):
    """
    消融A用：不做LLM验证，直接用含目标词的句子（不截断，按数量填充元信息）
    """
    candidates = [
        s for s in raw_sentences
        if target_word in (s["sentence"] if isinstance(s, dict) else str(s))
    ]
    valid_examples = [
        s["sentence"] if isinstance(s, dict) else str(s)
        for s in candidates
    ]

    if len(valid_examples) >= 3:
        confidence = "medium"
        evidence_status = "supported"
        contextual_fit = "medium"
        reason = f"检索到{len(valid_examples)}条含目标词的历史例句，证据充分"
    elif len(valid_examples) >= 1:
        confidence = "low"
        evidence_status = "supported"
        contextual_fit = "low"
        reason = f"检索到{len(valid_examples)}条含目标词的历史例句，证据较少"
    else:
        confidence = "unknown"
        evidence_status = "unknown"
        contextual_fit = "unknown"
        reason = "未检索到含目标词的历史例句；这表示证据缺失，不表示该义项不符合目标句"

    return {
        "wsid": sense_wsid,
        "gloss": sense_gloss,
        "valid_examples": valid_examples,
        "evidence_status": evidence_status,
        "contextual_fit": contextual_fit,
        "support": None,
        "confidence": confidence,
        "reason": reason
    }


def aggregator(text, target_word, agent_results, call_llm):
    """
    聚合器：综合所有义项智能体的判断，做最终词义消歧
    对应MADAM-RAG里聚合器统观全局做最终裁决的设计
    核心原则：证据缺失不等于义项错误，以语境契合度为准
    """
    # 判断是否所有义项都没有有效证据
    all_no_evidence = all(
        len(a["valid_examples"]) == 0
        for a in agent_results
    )

    # 构建证据对比块
    evidence_block = ""
    for i, agent in enumerate(agent_results):
        evidence_block += f"\n【候选义项{i+1}】"
        evidence_block += f"\nwsid: {agent['wsid']}"
        evidence_block += f"\n义项定义: {agent['gloss']}"
        evidence_block += f"\n证据状态: {agent.get('evidence_status', 'unknown')}"
        evidence_block += f"\n目标语境契合度: {agent.get('contextual_fit', 'unknown')}"
        evidence_block += f"\n置信度: {agent.get('confidence', 'unknown')}"
        evidence_block += f"\n判断理由: {agent.get('reason', '')}"

        if agent["valid_examples"]:
            evidence_block += f"\n经验证的代表性例句:"
            for sent in agent["valid_examples"][:2]:
                evidence_block += f"\n  · {sent}"
        else:
            evidence_block += f"\n经验证的代表性例句: 无（证据缺失，不作为排除理由）"
        evidence_block += "\n"

    if all_no_evidence:
        # 所有义项都没有有效证据 → 降级，不依赖历史证据
        prompt = f"""你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

历史证据检索未能找到有效例句。
请仅根据目标句的语境和以下义项定义进行判断。

候选义项：
{chr(10).join(
    f"wsid={a['wsid']}: {a['gloss']}"
    for a in agent_results
)}

请选出目标句中「{target_word}」最符合的义项。

输出严格JSON：
{{"prediction": "wsid", "reason": "判断理由"}}"""

        response = call_llm(prompt)

        # 清理markdown包装
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            result = json.loads(response)
            return (
                str(result["prediction"]),
                result.get("reason", ""),
                "降级模式：无历史证据，仅凭目标句和义项定义判断",
                "N/A"
            )
        except Exception as e:
            return (
                agent_results[0]["wsid"],
                f"解析失败: {e}",
                "N/A",
                "N/A"
            )
    else:
        # 有有效证据 → 用语境对比做消歧
        prompt = f"""你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

各候选义项的历史证据和语境判断如下：
{evidence_block}

请严格按以下顺序完成词义消歧：

第一步：Context-first 判断
仅根据目标句语境和义项定义，判断每个义项与目标句的契合度。
不要参考历史例句数量，也不要因为某义项没有历史例句而降低其合理性。

第二步：Evidence calibration
只检查历史例句是否与目标句语境高度同构。
历史证据只能作为辅助校准，不能覆盖第一步的语境判断。
有历史例句但语境不匹配的义项，不得加分。
无历史例句的义项，证据状态为 unknown，不等于错误或不支持。

第三步：Evidence-bias check
检查你是否因为某义项有历史例句而倾向选择它。
如果去掉历史例句后，另一个义项更符合目标句，则应优先选择语境契合度更高的义项。

重要原则：
1. Missing Evidence ≠ Rejection。
2. No evidence = Unknown, not Unsupported。
3. Evidence count is not probability。
4. Contextual compatibility has priority over historical evidence availability。
5. 如果某义项无历史证据但最符合目标句，应选择该义项。
6. 如果某义项有历史证据但与目标句语境不契合，不应选择该义项。

输出严格JSON：
{{
  "step1": "仅基于目标句和义项定义的语境判断",
  "step2": "历史证据是否真正与目标句语境同构",
  "bias_check": "是否存在因有证据而偏向某义项的风险",
  "prediction": "wsid",
  "reason": "最终判断理由"
}}"""

        response = call_llm(prompt)

        # 清理markdown包装
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            result = json.loads(response)
            initial_prediction = str(result["prediction"])
            step1_analysis = result.get("step1", "N/A")
            bias_check = result.get("bias_check", "N/A")
            step2_analysis = f"{result.get('step2', 'N/A')} | bias_check: {bias_check}"

            # 显式反向验证：检查是否因"证据缺失"而排除了某义项
            no_evidence_agents = [
                a for a in agent_results
                if len(a["valid_examples"]) == 0 and a["wsid"] != initial_prediction
            ]

            if no_evidence_agents:
                # 构建反向验证prompt
                excluded_block = ""
                for a in no_evidence_agents:
                    excluded_block += f"\n- wsid={a['wsid']}: {a['gloss']}（无历史证据）"

                verification_prompt = f"""你是古汉语词义消歧专家。

你已初步选择义项 {initial_prediction}。

目标句：{text}
目标词：「{target_word}」

以下义项没有历史证据，可能被排除：
{excluded_block}

请回答：
1. 你排除了上述哪些义项？
2. 排除它们的理由是什么？（是否仅因为它们缺乏历史证据？）

如果你的排除理由包含"缺乏历史证据"或类似表述，请重新评估。
重新思考：仅凭目标句语境和义项定义，这些被排除的义项是否可能更符合？

重要原则：证据缺失不等于义项错误。必须仅凭语境契合度判断。

输出严格JSON：
{{
  "excluded_check": "你是否仅因缺乏证据而排除了某些义项？",
  "reassessment": "重新评估后的判断",
  "final_prediction": "wsid（如果与初步选择不同则更新）",
  "final_reason": "最终判断理由"
}}"""

                verification_response = call_llm(verification_prompt)

                # 清理markdown包装
                verification_response = verification_response.strip()
                if verification_response.startswith("```json"):
                    verification_response = verification_response[7:]
                if verification_response.startswith("```"):
                    verification_response = verification_response[3:]
                if verification_response.endswith("```"):
                    verification_response = verification_response[:-3]
                verification_response = verification_response.strip()

                try:
                    verify_result = json.loads(verification_response)
                    final_prediction = str(verify_result.get("final_prediction", initial_prediction))
                    final_reason = verify_result.get("final_reason", result.get("reason", ""))
                    step1_analysis = f"{step1_analysis} | 验证: {verify_result.get('excluded_check', 'N/A')}"
                    step2_analysis = f"{step2_analysis} | 复审: {verify_result.get('reassessment', 'N/A')}"

                    return (
                        final_prediction,
                        final_reason,
                        step1_analysis,
                        step2_analysis
                    )
                except Exception:
                    # 验证解析失败，保持初步选择
                    return (
                        initial_prediction,
                        result.get("reason", ""),
                        step1_analysis,
                        step2_analysis
                    )
            else:
                # 没有需要验证的，保持初步选择
                return (
                    initial_prediction,
                    result.get("reason", ""),
                    step1_analysis,
                    step2_analysis
                )

        except Exception as e:
            # 解析失败时回退到第一个义项
            return (
                agent_results[0]["wsid"],
                f"解析失败: {e}",
                "N/A",
                "N/A"
            )


def vote_decision(agent_results):
    """
    消融B用：不用聚合器，改用置信度投票
    """
    conf_order = {"high": 3, "medium": 2, "low": 1, "none": 0, "unknown": 0}
    supporting = [a for a in agent_results if a.get("support") == True]

    if supporting:
        best = max(
            supporting,
            key=lambda a: conf_order.get(a.get("confidence", "none"), 0)
        )
        return (
            best["wsid"],
            f"投票：置信度最高的支持义项（{best.get('confidence', 'unknown')}）",
            "投票模式：无聚合器",
            "N/A"
        )
    else:
        return (
            agent_results[0]["wsid"],
            "无支持义项，默认选第一个候选",
            "投票模式：无支持义项",
            "N/A"
        )


def generate_topk_union(text, candidate_senses, k=2):
    """
    Generate Top-k union using LLM (three judgments and take union)
    This is a COMMON pre-processing step shared by both configurations

    Args:
        text (str): Current context text
        candidate_senses (list): List of candidate senses
        k (int): Number of senses to return

    Returns:
        tuple: (union_senses, vote_scores, run_results) - Union senses list, vote scores, and individual run results
    """
    # Three judgments and take union
    all_wsids = set()
    # Record vote counts for each sense
    vote_counts = defaultdict(int)
    run_results = []

    for i in range(3):
        print(f"[LLM Judgment #{i+1}]")
        # Build prompt
        sense_block = "\n".join(
            f"[{idx+1}] wsid={s['wsid']}: {s['newgloss']}"
            for idx, s in enumerate(candidate_senses)
        )

        prompt = f"""
You are performing an [Ancient Chinese Word Sense Disambiguation Task].

[Target Sentence]
{text}

[Candidate Senses]
{sense_block}

Please complete the following tasks:
1. Select the [most likely {k} senses] from the above options;
2. No final judgment required, only need to be "reasonable in the current context";
3. Provide a brief reason for each selected sense.

[Output Format (strict JSON)]
{{"top_senses": [{{"wsid": "...", "reason": "..."}}]}}
"""

        # Extract target word from candidate senses
        target_word = candidate_senses[0]['word'] if candidate_senses else "unknown"
        # Call LLM with cache - include context in cache tag to avoid contamination
        context_hash = hash(text[:100]) % 1000000  # Simple hash to avoid long cache tags
        cache_tag = f"topk|word={target_word}|context={context_hash}|run={i}"
        response = cached_call_llm(prompt, cache_tag=cache_tag, seed=42 + i)

        # Remove markdown code block markers (if present)
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()

        try:
            # Parse JSON
            result = json.loads(response)
            # Extract wsid of Top-k senses
            run_wsids = []
            for item in result.get("top_senses", []):
                wsid = str(item["wsid"])
                all_wsids.add(wsid)
                vote_counts[wsid] += 1
                run_wsids.append(wsid)
            print(f"[LLM Judgment #{i+1}] Successfully extracted {len(result.get('top_senses', []))} senses")
            run_results.append(run_wsids)
        except Exception as e:
            print(f"[LLM Judgment #{i+1}] Failed to parse response: {e}")
            run_results.append([])
            continue

    print(f"[Three judgments result] Union size: {len(all_wsids)}")
    # Convert to sense list
    union_senses = [s for s in candidate_senses if str(s["wsid"]) in all_wsids]

    # Calculate vote scores (normalized to 0-1)
    vote_scores = {}
    for sense in candidate_senses:
        wsid = str(sense["wsid"])
        # Vote score = number of votes / 3
        vote_scores[wsid] = vote_counts.get(wsid, 0) / 3

    return union_senses, vote_scores, run_results


def process_sample_with_union(text, candidate_senses, gold_wsid, gold_gloss, top2_senses,
                              wsid_to_context, wsid_to_sense, test_word_to_senses,
                              sense_embedding_cache, preloaded_corpus, config=None):
    """
    Process a single test sample with pre-generated Top-k union

    Args:
        text (str): Target sentence
        candidate_senses (list): All candidate senses for this word
        gold_wsid (str): Gold standard wsid
        gold_gloss (str): Gold standard gloss
        top2_senses (list): Pre-generated Top-2 union senses (shared by both configurations)
        wsid_to_context (dict): Mapping from wsid to context
        wsid_to_sense (dict): Mapping from wsid to sense
        test_word_to_senses (dict): Mapping from test word to senses
        sense_embedding_cache (dict): Sense embedding cache
        preloaded_corpus (list): Preloaded corpus
        config (dict): Configuration options

    Returns:
        dict: Detailed result for this sample
    """
    # Use default configuration
    if config is None:
        config = BASE_CONFIG

    word = candidate_senses[0]['word'] if candidate_senses else "unknown"

    # Determine which senses to use based on configuration
    if config.get("use_top2_filter", True):
        # Use Top-k union senses
        senses_to_use = top2_senses
        topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
        topk_union_glosses = [s.get("newgloss", "") for s in top2_senses]
        gold_in_topk_union = gold_wsid in topk_union_wsids
    else:
        # Pure LLM: Use all candidate senses directly
        senses_to_use = candidate_senses
        topk_union_wsids = [str(s["wsid"]) for s in candidate_senses]
        topk_union_glosses = [s.get("newgloss", "") for s in candidate_senses]
        gold_in_topk_union = gold_wsid in topk_union_wsids

    # Build historical evidence with MADAM-RAG architecture
    agent_results = []
    agent_raw_counts = []

    for sense in senses_to_use:
        sense_wsid = str(sense["wsid"])
        sense_gloss = sense.get("newgloss", "")

        raw_sentences = []
        if config["use_supporting_sentences"]:
            try:
                index_dir = str(BASE_DIR / "bm25_index_zh")
                raw_sentences = get_supporting_sentences_with_pseudo_bm25(
                    sense_gloss=sense_gloss,
                    target_word=word,
                    index_dir=index_dir,
                    wsid=sense_wsid,
                    bm25_top_k=200,
                    final_top_k=10,
                    threshold=0.5
                )
            except Exception:
                raw_sentences = []

            # 根据配置选择是否做LLM验证
            if config.get("use_agent_verify", True):
                agent_output = sense_agent(
                    text=text,
                    target_word=word,
                    sense_gloss=sense_gloss,
                    sense_wsid=sense_wsid,
                    raw_sentences=raw_sentences,
                    cached_call_llm=cached_call_llm
                )
            else:
                # 消融A：不做验证
                agent_output = sense_agent_no_verify(
                    text=text,
                    target_word=word,
                    sense_gloss=sense_gloss,
                    sense_wsid=sense_wsid,
                    raw_sentences=raw_sentences
                )
            agent_results.append(agent_output)
            # Count raw sentences that contain target word
            raw_count = len([s for s in raw_sentences if word in (s["sentence"] if isinstance(s, dict) else str(s))])
            agent_raw_counts.append(raw_count)
        else:
            # No supporting sentences, create empty agent result
            agent_results.append({
                "wsid": sense_wsid,
                "gloss": sense_gloss,
                "valid_examples": [],
                "evidence_status": "unknown",
                "contextual_fit": "unknown",
                "support": None,
                "confidence": "unknown",
                "reason": "未启用支持句检索"
            })
            agent_raw_counts.append(0)

    # Final decision
    prediction = ""
    predicted_gloss = ""
    is_correct = False
    excluded = []
    reason = ""

    # Build compatible sense_evidence_list for backward compatibility
    sense_evidence_list = []
    for agent in agent_results:
        # Convert agent results to compatible format
        sense_evidence_list.append({
            "wsid": agent["wsid"],
            "gloss": agent["gloss"],
            "prototype_vector": None,
            "supporting_sentences": [{"sentence": s} for s in agent["valid_examples"]],
            "similarity": 0.0
        })

    if config["final_decision_with_llm"]:
        try:
            # 根据配置选择是否用聚合器
            if config.get("use_aggregator", True):
                prediction, reason, aggregator_step1, aggregator_step2 = aggregator(
                    text=text,
                    target_word=word,
                    agent_results=agent_results,
                    call_llm=call_llm
                )
            else:
                # 消融B：用投票代替聚合器
                prediction, reason, aggregator_step1, aggregator_step2 = vote_decision(
                    agent_results
                )

            # Get predicted gloss
            for sense in candidate_senses:
                if str(sense["wsid"]) == prediction:
                    predicted_gloss = sense.get("newgloss", "")
                    break

            # Verify prediction result
            is_correct = prediction == gold_wsid

        except Exception as e:
            reason = f"Error: {str(e)}"
            aggregator_step1 = "N/A"
            aggregator_step2 = "N/A"
    else:
        # Not using LLM for final decision, use first sense
        if sense_evidence_list:
            # Use first sense
            first_evidence = sense_evidence_list[0]
            prediction = first_evidence['wsid']
            predicted_gloss = first_evidence['gloss']

            # Verify prediction result
            is_correct = prediction == gold_wsid
            reason = "Using first candidate sense (LLM decision disabled)"
        else:
            reason = "No candidate senses"
        aggregator_step1 = "N/A"
        aggregator_step2 = "N/A"

    # Collect supporting sentences text (per sense, unique, up to 10 per sense)
    supporting_sentences_text = []
    for idx, evidence in enumerate(sense_evidence_list):
        sense_label = f"【义项{idx+1}：{evidence['gloss']}】"
        
        if evidence['supporting_sentences']:
            seen = set()
            unique_sents = []
            for sent in evidence['supporting_sentences']:
                if sent['sentence'] not in seen:
                    seen.add(sent['sentence'])
                    unique_sents.append(sent['sentence'])
                    if len(unique_sents) >= 10:
                        break
            
            if unique_sents:
                # Format each sentence with numbering
                formatted_sents = []
                for sent_idx, sent in enumerate(unique_sents, 1):
                    formatted_sents.append(f"{sent_idx}. {sent}")
                sent_text = "\n".join(formatted_sents)
                supporting_sentences_text.append(f"{sense_label}\n历史例句:\n{sent_text}")
            else:
                supporting_sentences_text.append(f"{sense_label}\n历史例句: [暂无直接历史例句，请依据目标句与义项定义判断]")
        else:
            supporting_sentences_text.append(f"{sense_label}\n历史例句: [暂无直接历史例句，请依据目标句与义项定义判断]")

    # Collect similarity scores
    similarity_scores = {}
    for evidence in sense_evidence_list:
        similarity_scores[evidence['wsid']] = evidence['similarity']

    # Build result
    result = {
        'all_candidate_wsids': ', '.join([str(s["wsid"]) for s in candidate_senses]),
        'all_candidate_glosses': ', '.join([s.get("newgloss", "") for s in candidate_senses]),
        'topk_union_wsids': ', '.join(topk_union_wsids),
        'topk_union_glosses': ', '.join(topk_union_glosses),
        'gold_in_topk_union': 'yes' if gold_in_topk_union else 'no',
        'prediction': prediction,
        'predicted_gloss': predicted_gloss,
        'is_correct': is_correct,
        'excluded': ', '.join(excluded),
        'reason': reason,
        'supporting_sentences': ' | '.join(supporting_sentences_text),
        'similarity': json.dumps(similarity_scores, ensure_ascii=False),
        # 新增：义项智能体层面字段
        'agent_raw_count': sum(agent_raw_counts),
        'agent_valid_count': sum(len(a["valid_examples"]) for a in agent_results),
        'agent_filter_rate': round(1 - sum(len(a["valid_examples"]) for a in agent_results) / sum(agent_raw_counts), 2) if sum(agent_raw_counts) > 0 else 1.0,
        'agent_support_pattern': '|'.join(str(a["support"]).lower() for a in agent_results),
        'agent_all_no_evidence': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'agent_details': json.dumps([
            {
                "wsid": a["wsid"],
                "gloss": a["gloss"],
                "raw_count": agent_raw_counts[i] if i < len(agent_raw_counts) else 0,
                "valid_count": len(a["valid_examples"]),
                "valid_examples": a["valid_examples"],
                "support": a["support"],
                "confidence": a["confidence"],
                "reason": a["reason"]
            }
            for i, a in enumerate(agent_results)
        ], ensure_ascii=False),
        # 新增：聚合器层面字段
        'aggregator_degraded': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'aggregator_step1': aggregator_step1,
        'aggregator_step2': aggregator_step2
    }

    return result


def run_comparison_test(num_rounds=2, num_samples_per_round=20):
    """
    Run comparison test between full and ablation_no_history configurations
    with paired samples and perform t-test

    Args:
        num_rounds (int): Number of rounds (default: 2)
        num_samples_per_round (int): Number of samples per round (default: 20)
    """
    print("=" * 90)
    print("Comparison Test: full vs ablation_no_history (Local version)")
    print("=" * 90)
    print(f"Experiment setup: {num_rounds} rounds × {num_samples_per_round} samples per round")
    print("Same samples used for both configurations in each round")
    print("Same Top-k union generated once, shared by both configurations")
    print("Updated: Using historical usage summary instead of schema")
    print("Added: candidate_count, pairwise similarity metrics, case_type classification")
    print("=" * 90)

    # Load test data
    contexts, all_senses, wsid_to_context, wsid_to_sense, all_word_to_senses, preloaded_corpus = load_test_data()

    # Process contexts: filter out contexts with only 1 sense
    filtered_contexts = []
    for ctx in contexts:
        wsid = str(ctx["wsid"])
        sense = wsid_to_sense.get(wsid)
        if sense:
            word = sense["word"]
            # Check if this word has more than 1 sense
            if len(all_word_to_senses.get(word, [])) > 1:
                filtered_contexts.append(ctx)

    print(f"Filtered contexts (with >1 sense): {len(filtered_contexts)}")

    # Initialize results
    all_results = []
    round_results = []

    # Run specified number of rounds
    for round_num in range(num_rounds):
        print(f"\n{'=' * 90}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'=' * 90}")

        # Randomly select samples for this round
        sampled_contexts = random.sample(filtered_contexts, min(num_samples_per_round, len(filtered_contexts)))
        print(f"Selected {len(sampled_contexts)} samples for this round")

        # Convert contexts to senses
        sampled_senses = []
        for ctx in sampled_contexts:
            wsid = str(ctx["wsid"])
            sense = wsid_to_sense.get(wsid)
            if sense:
                sampled_senses.append(sense)

        # Extract target word set for testing
        test_words = set()
        for sense in sampled_senses:
            test_words.add(sense["word"])

        # Build test word_to_senses mapping
        test_word_to_senses = {}
        for word in test_words:
            test_word_to_senses[word] = all_word_to_senses[word]

        # Initialize empty sense_embedding_cache
        sense_embedding_cache = {}

        # Process each sample for all configurations
        full_correct = 0
        no_history_correct = 0
        pure_llm_correct = 0
        no_verify_correct = 0  # 消融A
        no_aggregator_correct = 0  # 消融B

        for sample_idx, sample in enumerate(sampled_senses):
            print(f"\nProcessing sample {sample_idx + 1}/{len(sampled_senses)}")

            wsid = str(sample["wsid"])
            word = sample["word"]
            context = wsid_to_context.get(wsid)
            if not context:
                continue

            text = context["txt"]
            gold_wsid = wsid
            gold_gloss = sample.get("newgloss", "")

            # Get all candidate senses for this word
            candidate_senses = test_word_to_senses.get(word, [])
            if not candidate_senses:
                continue

            # Calculate similarity metrics for candidate senses
            similarity_metrics = calculate_sense_similarity_metrics(candidate_senses)
            candidate_count = similarity_metrics['candidate_count']
            mean_pairwise_similarity = similarity_metrics['mean_pairwise_similarity']
            max_pairwise_similarity = similarity_metrics['max_pairwise_similarity']
            top2_similarity = similarity_metrics['top2_similarity']

            # =====================================================================
            # KEY IMPROVEMENT: Generate Top-k union ONCE, shared by both configurations
            # =====================================================================
            print("  [COMMON] Generating Top-k union (shared by both configurations)...")
            top2_senses, vote_scores, run_results = generate_topk_union(text, candidate_senses, k=2)
            print(f"  [COMMON] Top-k union generated: {len(top2_senses)} senses")

            # Calculate similarity metrics for Top-k union senses
            topk_similarity_metrics = calculate_sense_similarity_metrics(top2_senses)
            topk_candidate_count = topk_similarity_metrics['candidate_count']
            topk_mean_pairwise_similarity = topk_similarity_metrics['mean_pairwise_similarity']
            topk_max_pairwise_similarity = topk_similarity_metrics['max_pairwise_similarity']
            topk_top2_similarity = topk_similarity_metrics['top2_similarity']

            # Check if gold is in topk union
            topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
            gold_in_topk_union = gold_wsid in topk_union_wsids

            # Test all configurations with random order to avoid order bias
            config_order = [
                "full",
                "ablation_no_history",
                "pure_llm",
                "ablation_no_verify",      # 消融A
                "ablation_no_aggregator"   # 消融B
            ]
            random.shuffle(config_order)
            results_by_config = {}

            for config_name in config_order:
                print(f"  Testing {config_name} configuration...")
                top2_input = candidate_senses if config_name == "pure_llm" else top2_senses

                result = process_sample_with_union(
                    text=text,
                    candidate_senses=candidate_senses,
                    gold_wsid=gold_wsid,
                    gold_gloss=gold_gloss,
                    top2_senses=top2_input,
                    wsid_to_context=wsid_to_context,
                    wsid_to_sense=wsid_to_sense,
                    test_word_to_senses=test_word_to_senses,
                    sense_embedding_cache=sense_embedding_cache,
                    preloaded_corpus=preloaded_corpus,
                    config=CONFIGS[config_name]
                )

                results_by_config[config_name] = result

            # Extract results from dictionary (order-independent)
            full_result = results_by_config["full"]
            no_history_result = results_by_config["ablation_no_history"]
            pure_llm_result = results_by_config["pure_llm"]

            # Count correct predictions
            full_is_correct = full_result.get('is_correct', False)
            if full_is_correct:
                full_correct += 1

            no_history_is_correct = no_history_result.get('is_correct', False)
            if no_history_is_correct:
                no_history_correct += 1

            pure_llm_is_correct = pure_llm_result.get('is_correct', False)
            if pure_llm_is_correct:
                pure_llm_correct += 1

            # 消融A
            no_verify_result = results_by_config.get("ablation_no_verify", {})
            no_verify_is_correct = no_verify_result.get('is_correct', False)
            if no_verify_is_correct:
                no_verify_correct += 1

            # 消融B
            no_aggregator_result = results_by_config.get("ablation_no_aggregator", {})
            no_aggregator_is_correct = no_aggregator_result.get('is_correct', False)
            if no_aggregator_is_correct:
                no_aggregator_correct += 1

            # Determine case type
            case_type = determine_case_type(full_is_correct, no_history_is_correct)

            # Analyze history quality from full configuration
            history_quality_label = analyze_history_quality_v2(full_result.get('agent_details', '[]'))

            # Build merged sample result (all configurations in one row)
            merged_result = {
                'round': round_num + 1,
                'sample_id': sample_idx + 1,
                'word': word,
                'context': text,
                'gold_wsid': gold_wsid,
                'gold_gloss': gold_gloss,
                'candidate_count': candidate_count,
                'mean_pairwise_similarity': mean_pairwise_similarity,
                'max_pairwise_similarity': max_pairwise_similarity,
                'top2_similarity': top2_similarity,
                'topk_candidate_count': topk_candidate_count,
                'topk_mean_pairwise_similarity': topk_mean_pairwise_similarity,
                'topk_max_pairwise_similarity': topk_max_pairwise_similarity,
                'topk_top2_similarity': topk_top2_similarity,
                'gold_in_topk_union': 'yes' if gold_in_topk_union else 'no',
                'onlyllm_correct': pure_llm_is_correct,
                'nohistory_correct': no_history_is_correct,
                'full_correct': full_is_correct,
                'case_type': case_type,
                'history_quality_label': history_quality_label,
                # Full configuration details
                'full_prediction': full_result.get('prediction', ''),
                'full_predicted_gloss': full_result.get('predicted_gloss', ''),
                'full_reason': full_result.get('reason', ''),
                'full_supporting_sentences': full_result.get('supporting_sentences', ''),
                # No history configuration details
                'nohistory_prediction': no_history_result.get('prediction', ''),
                'nohistory_predicted_gloss': no_history_result.get('predicted_gloss', ''),
                'nohistory_reason': no_history_result.get('reason', ''),
                # Pure LLM configuration details
                'purellm_prediction': pure_llm_result.get('prediction', ''),
                'purellm_predicted_gloss': pure_llm_result.get('predicted_gloss', ''),
                'purellm_reason': pure_llm_result.get('reason', ''),
                # 消融A配置详情
                'no_verify_correct': no_verify_is_correct,
                'no_verify_prediction': no_verify_result.get('prediction', ''),
                # 消融B配置详情
                'no_aggregator_correct': no_aggregator_is_correct,
                'no_aggregator_prediction': no_aggregator_result.get('prediction', ''),
                # 新增：义项智能体层面字段
                'agent_raw_count': full_result.get('agent_raw_count', 0),
                'agent_valid_count': full_result.get('agent_valid_count', 0),
                'agent_filter_rate': full_result.get('agent_filter_rate', 1.0),
                'agent_support_pattern': full_result.get('agent_support_pattern', ''),
                'agent_all_no_evidence': full_result.get('agent_all_no_evidence', False),
                'agent_details': full_result.get('agent_details', ''),
                # 新增：聚合器层面字段
                'aggregator_degraded': full_result.get('aggregator_degraded', False),
                'aggregator_step1': full_result.get('aggregator_step1', ''),
                'aggregator_step2': full_result.get('aggregator_step2', '')
            }

            all_results.append(merged_result)

        # Calculate round accuracy
        full_accuracy = full_correct / len(sampled_senses) if sampled_senses else 0.0
        no_history_accuracy = no_history_correct / len(sampled_senses) if sampled_senses else 0.0
        pure_llm_accuracy = pure_llm_correct / len(sampled_senses) if sampled_senses else 0.0

        print(f"\nRound {round_num + 1} results:")
        print(f"  Full configuration: {full_correct}/{len(sampled_senses)} = {full_accuracy:.3f}")
        print(f"  Ablation_no_history: {no_history_correct}/{len(sampled_senses)} = {no_history_accuracy:.3f}")
        print(f"  Pure LLM: {pure_llm_correct}/{len(sampled_senses)} = {pure_llm_accuracy:.3f}")

        # Save round results
        round_results.append({
            'round': round_num + 1,
            'full_accuracy': full_accuracy,
            'no_history_accuracy': no_history_accuracy,
            'pure_llm_accuracy': pure_llm_accuracy,
            'no_verify_accuracy': no_verify_correct / len(sampled_senses) if sampled_senses else 0,
            'no_aggregator_accuracy': no_aggregator_correct / len(sampled_senses) if sampled_senses else 0,
            'difference': no_history_accuracy - full_accuracy
        })

    # Convert round results to DataFrame
    round_df = pd.DataFrame(round_results)
    print(f"\n{'=' * 90}")
    print("Round Results Summary:")
    print(f"{'=' * 90}")
    print(round_df)

    # Perform paired t-test
    print(f"\n{'=' * 90}")
    print("Paired t-test Results:")
    print(f"{'=' * 90}")

    # Extract paired observations
    full_scores = round_df['full_accuracy'].tolist()
    no_history_scores = round_df['no_history_accuracy'].tolist()
    differences = round_df['difference'].tolist()

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(no_history_scores, full_scores)

    # Calculate descriptive statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    n = len(differences)

    # Calculate 95% confidence interval
    if n > 1 and std_diff > 0:
        ci = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))
    else:
        ci = (np.nan, np.nan)

    # Print results
    print(f"Mean difference (no_history - full): {mean_diff:.4f}")
    print(f"Standard deviation: {std_diff:.4f}")
    print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Interpret results
    alpha = 0.05
    if not np.isnan(p_value):
        if p_value < alpha:
            print("\n✅ Significant difference found (p < 0.05)")
            if mean_diff > 0:
                print("   ablation_no_history performs better than full")
            else:
                print("   full performs better than ablation_no_history")
        else:
            print("\n❌ No significant difference found (p >= 0.05)")
    else:
        print("\n⚠️  Cannot perform t-test (need at least 2 rounds)")

    # Export detailed results
    print(f"\n{'=' * 90}")
    print("Exporting detailed results...")
    print(f"{'=' * 90}")

    try:
        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Convert all results to DataFrame
        detail_df = pd.DataFrame(all_results)

        # Define column order
        columns_order = [
            'round', 'sample_id', 'word', 'context', 'gold_wsid', 'gold_gloss',
            'candidate_count', 'mean_pairwise_similarity', 'max_pairwise_similarity', 'top2_similarity',
            'topk_candidate_count', 'topk_mean_pairwise_similarity', 'topk_max_pairwise_similarity', 'topk_top2_similarity',
            'gold_in_topk_union', 'onlyllm_correct', 'nohistory_correct', 'full_correct',
            'case_type', 'history_quality_label',
            'full_prediction', 'full_predicted_gloss', 'full_reason', 'full_supporting_sentences',
            'nohistory_prediction', 'nohistory_predicted_gloss', 'nohistory_reason',
            'purellm_prediction', 'purellm_predicted_gloss', 'purellm_reason',
            # 消融配置详情
            'no_verify_correct', 'no_aggregator_correct',
            'no_verify_prediction', 'no_aggregator_prediction',
            # 新增：义项智能体层面字段
            'agent_raw_count', 'agent_valid_count', 'agent_filter_rate',
            'agent_support_pattern', 'agent_all_no_evidence', 'agent_details',
            # 新增：聚合器层面字段
            'aggregator_degraded', 'aggregator_step1', 'aggregator_step2'
        ]
        detail_df = detail_df[columns_order]

        # Export to Excel
        excel_file = f'local_comparison_full_vs_nohistory_detailed_{timestamp}.xlsx'
        detail_df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Successfully exported detailed results to {excel_file}")

        # Export to CSV
        csv_file = f'local_comparison_full_vs_nohistory_detailed_{timestamp}.csv'
        detail_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ Successfully exported detailed results to {csv_file}")

        # Export round results
        round_excel_file = f'local_comparison_full_vs_nohistory_rounds_{timestamp}.xlsx'
        round_df.to_excel(round_excel_file, index=False, engine='openpyxl')
        print(f"✅ Successfully exported round results to {round_excel_file}")

    except Exception as e:
        print(f"⚠️  Failed to export results: {e}")

    return round_results


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare full vs ablation_no_history configurations with paired t-test')
    parser.add_argument('--rounds', type=int, default=2, help='Number of rounds (default: 2)')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples per round (default: 20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run comparison test with specified parameters
    run_comparison_test(num_rounds=args.rounds, num_samples_per_round=args.samples)
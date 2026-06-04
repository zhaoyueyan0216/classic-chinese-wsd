# Comparison test for full vs ablation_no_history configurations with paired t-test (Local version)
# Key improvement: Top-k union is generated once per sample, then shared by both configurations
# Updated: Using no-pseudo retrieval (direct gloss search)
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
from core.pseudo_retrieval import get_supporting_sentences_without_pseudo_bm25
from core.llm import call_llm, cached_call_llm


# Base configuration
BASE_CONFIG = {
    "use_top2_filter": True,
    "use_pseudo_query": False,  # No pseudo query - use gloss directly
    "use_bm25": True,
    "use_rerank": True,
    "use_schema": False,
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
CONFIGS["pure_llm"]["use_top2_filter"] = False
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

    preloaded_corpus = []

    return contexts, senses, wsid_to_context, wsid_to_sense, word_to_senses, preloaded_corpus


def calculate_sense_similarity_metrics(candidate_senses):
    """
    Calculate pairwise similarity metrics for candidate senses
    """
    candidate_count = len(candidate_senses)
    
    if candidate_count < 2:
        return {
            'candidate_count': candidate_count,
            'mean_pairwise_similarity': 0.0,
            'max_pairwise_similarity': 0.0,
            'top2_similarity': 0.0
        }
    
    glosses = [s.get("newgloss", "") for s in candidate_senses]
    embeddings = [get_embedding(gloss) for gloss in glosses]
    
    similarities = []
    for i in range(candidate_count):
        for j in range(i + 1, candidate_count):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
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
    if full_correct and not no_history_correct:
        return 'A'
    elif not full_correct and no_history_correct:
        return 'B'
    elif full_correct and no_history_correct:
        return 'C'
    else:
        return 'D'


def analyze_history_quality_v2(agent_details_str):
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
    candidates = [
        s for s in raw_sentences
        if target_word in (s["sentence"] if isinstance(s, dict) else str(s))
    ]

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
    candidates = [
        s for s in raw_sentences
        if target_word in (s["sentence"] if isinstance(s, dict) else str(s))
    ]
    return {
        "wsid": sense_wsid,
        "gloss": sense_gloss,
        "valid_examples": [
            s["sentence"] if isinstance(s, dict) else str(s)
            for s in candidates[:3]
        ],
        "evidence_status": "unknown",
        "contextual_fit": "unknown",
        "support": len(candidates) > 0,
        "confidence": "medium" if candidates else "unknown",
        "reason": "未经LLM验证，直接使用含目标词的句子"
    }


def aggregator(text, target_word, agent_results, call_llm):
    all_no_evidence = all(
        len(a["valid_examples"]) == 0
        for a in agent_results
    )

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

            no_evidence_agents = [
                a for a in agent_results
                if len(a["valid_examples"]) == 0 and a["wsid"] != initial_prediction
            ]

            if no_evidence_agents:
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
                    return (
                        initial_prediction,
                        result.get("reason", ""),
                        step1_analysis,
                        step2_analysis
                    )
            else:
                return (
                    initial_prediction,
                    result.get("reason", ""),
                    step1_analysis,
                    step2_analysis
                )

        except Exception as e:
            return (
                agent_results[0]["wsid"],
                f"解析失败: {e}",
                "N/A",
                "N/A"
            )


def vote_decision(agent_results):
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
    all_wsids = set()
    vote_counts = defaultdict(int)
    run_results = []

    for i in range(3):
        print(f"[LLM Judgment #{i+1}]")
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

        target_word = candidate_senses[0]['word'] if candidate_senses else "unknown"
        context_hash = hash(text[:100]) % 1000000
        cache_tag = f"topk|word={target_word}|context={context_hash}|run={i}"
        response = cached_call_llm(prompt, cache_tag=cache_tag, seed=42 + i)

        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()

        try:
            result = json.loads(response)
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
    union_senses = [s for s in candidate_senses if str(s["wsid"]) in all_wsids]

    vote_scores = {}
    for sense in candidate_senses:
        wsid = str(sense["wsid"])
        vote_scores[wsid] = vote_counts.get(wsid, 0) / 3

    return union_senses, vote_scores, run_results


def process_sample_with_union(text, candidate_senses, gold_wsid, gold_gloss, top2_senses,
                              wsid_to_context, wsid_to_sense, test_word_to_senses,
                              sense_embedding_cache, preloaded_corpus, config=None):
    if config is None:
        config = BASE_CONFIG

    word = candidate_senses[0]['word'] if candidate_senses else "unknown"

    if config.get("use_top2_filter", True):
        senses_to_use = top2_senses
        topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
        topk_union_glosses = [s.get("newgloss", "") for s in top2_senses]
        gold_in_topk_union = gold_wsid in topk_union_wsids
    else:
        senses_to_use = candidate_senses
        topk_union_wsids = [str(s["wsid"]) for s in candidate_senses]
        topk_union_glosses = [s.get("newgloss", "") for s in candidate_senses]
        gold_in_topk_union = gold_wsid in topk_union_wsids

    agent_results = []
    agent_raw_counts = []

    for sense in senses_to_use:
        sense_wsid = str(sense["wsid"])
        sense_gloss = sense.get("newgloss", "")

        raw_sentences = []
        if config["use_supporting_sentences"]:
            try:
                index_dir = str(BASE_DIR / "bm25_index_zh")
                # Use no-pseudo retrieval (direct gloss search)
                raw_sentences = get_supporting_sentences_without_pseudo_bm25(
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
                agent_output = sense_agent_no_verify(
                    text=text,
                    target_word=word,
                    sense_gloss=sense_gloss,
                    sense_wsid=sense_wsid,
                    raw_sentences=raw_sentences
                )
            agent_results.append(agent_output)
            raw_count = len([s for s in raw_sentences if word in (s["sentence"] if isinstance(s, dict) else str(s))])
            agent_raw_counts.append(raw_count)
        else:
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

    prediction = ""
    predicted_gloss = ""
    is_correct = False
    excluded = []
    reason = ""

    sense_evidence_list = []
    for agent in agent_results:
        sense_evidence_list.append({
            "wsid": agent["wsid"],
            "gloss": agent["gloss"],
            "prototype_vector": None,
            "supporting_sentences": [{"sentence": s} for s in agent["valid_examples"]],
            "similarity": 0.0
        })

    if config["final_decision_with_llm"]:
        try:
            if config.get("use_aggregator", True):
                prediction, reason, aggregator_step1, aggregator_step2 = aggregator(
                    text=text,
                    target_word=word,
                    agent_results=agent_results,
                    call_llm=call_llm
                )
            else:
                prediction, reason, aggregator_step1, aggregator_step2 = vote_decision(
                    agent_results
                )

            for sense in candidate_senses:
                if str(sense["wsid"]) == prediction:
                    predicted_gloss = sense.get("newgloss", "")
                    break

            is_correct = prediction == gold_wsid

        except Exception as e:
            reason = f"Error: {str(e)}"
            aggregator_step1 = "N/A"
            aggregator_step2 = "N/A"
    else:
        if sense_evidence_list:
            first_evidence = sense_evidence_list[0]
            prediction = first_evidence['wsid']
            predicted_gloss = first_evidence['gloss']
            is_correct = prediction == gold_wsid
            reason = "Using first candidate sense (LLM decision disabled)"
        else:
            reason = "No candidate senses"
        aggregator_step1 = "N/A"
        aggregator_step2 = "N/A"

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
                formatted_sents = []
                for sent_idx, sent in enumerate(unique_sents, 1):
                    formatted_sents.append(f"{sent_idx}. {sent}")
                sent_text = "\n".join(formatted_sents)
                supporting_sentences_text.append(f"{sense_label}\n历史例句:\n{sent_text}")
            else:
                supporting_sentences_text.append(f"{sense_label}\n历史例句: [暂无直接历史例句，请依据目标句与义项定义判断]")
        else:
            supporting_sentences_text.append(f"{sense_label}\n历史例句: [暂无直接历史例句，请依据目标句与义项定义判断]")

    similarity_scores = {}
    for evidence in sense_evidence_list:
        similarity_scores[evidence['wsid']] = evidence['similarity']

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
        'aggregator_degraded': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'aggregator_step1': aggregator_step1,
        'aggregator_step2': aggregator_step2
    }

    return result


def run_comparison_test(num_rounds=2, num_samples_per_round=20):
    print("=" * 90)
    print("Comparison Test: full vs ablation_no_history (Local version - NO PSEUDO)")
    print("=" * 90)
    print(f"Experiment setup: {num_rounds} rounds × {num_samples_per_round} samples per round")
    print("Same samples used for both configurations in each round")
    print("Same Top-k union generated once, shared by both configurations")
    print("KEY: Using no-pseudo retrieval (direct gloss search)")
    print("=" * 90)

    contexts, all_senses, wsid_to_context, wsid_to_sense, all_word_to_senses, preloaded_corpus = load_test_data()

    filtered_contexts = []
    for ctx in contexts:
        wsid = str(ctx["wsid"])
        sense = wsid_to_sense.get(wsid)
        if sense:
            word = sense["word"]
            if len(all_word_to_senses.get(word, [])) > 1:
                filtered_contexts.append(ctx)

    print(f"Filtered contexts (with >1 sense): {len(filtered_contexts)}")

    all_results = []
    round_results = []

    for round_num in range(num_rounds):
        print(f"\n{'=' * 90}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'=' * 90}")

        sampled_contexts = random.sample(filtered_contexts, min(num_samples_per_round, len(filtered_contexts)))
        print(f"Selected {len(sampled_contexts)} samples for this round")

        sampled_senses = []
        for ctx in sampled_contexts:
            wsid = str(ctx["wsid"])
            sense = wsid_to_sense.get(wsid)
            if sense:
                sampled_senses.append(sense)

        test_words = set()
        for sense in sampled_senses:
            test_words.add(sense["word"])

        test_word_to_senses = {}
        for word in test_words:
            test_word_to_senses[word] = all_word_to_senses[word]

        sense_embedding_cache = {}

        full_correct = 0
        no_history_correct = 0
        pure_llm_correct = 0
        no_verify_correct = 0
        no_aggregator_correct = 0

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

            candidate_senses = test_word_to_senses.get(word, [])
            if not candidate_senses:
                continue

            similarity_metrics = calculate_sense_similarity_metrics(candidate_senses)
            candidate_count = similarity_metrics['candidate_count']
            mean_pairwise_similarity = similarity_metrics['mean_pairwise_similarity']
            max_pairwise_similarity = similarity_metrics['max_pairwise_similarity']
            top2_similarity = similarity_metrics['top2_similarity']

            print("  [COMMON] Generating Top-k union (shared by both configurations)...")
            top2_senses, vote_scores, run_results = generate_topk_union(text, candidate_senses, k=2)
            print(f"  [COMMON] Top-k union generated: {len(top2_senses)} senses")

            topk_similarity_metrics = calculate_sense_similarity_metrics(top2_senses)
            topk_candidate_count = topk_similarity_metrics['candidate_count']
            topk_mean_pairwise_similarity = topk_similarity_metrics['mean_pairwise_similarity']
            topk_max_pairwise_similarity = topk_similarity_metrics['max_pairwise_similarity']
            topk_top2_similarity = topk_similarity_metrics['top2_similarity']

            topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
            gold_in_topk_union = gold_wsid in topk_union_wsids

            config_order = [
                "full",
                "ablation_no_history",
                "pure_llm",
                "ablation_no_verify",
                "ablation_no_aggregator"
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

            full_result = results_by_config["full"]
            no_history_result = results_by_config["ablation_no_history"]
            pure_llm_result = results_by_config["pure_llm"]

            full_is_correct = full_result.get('is_correct', False)
            if full_is_correct:
                full_correct += 1

            no_history_is_correct = no_history_result.get('is_correct', False)
            if no_history_is_correct:
                no_history_correct += 1

            pure_llm_is_correct = pure_llm_result.get('is_correct', False)
            if pure_llm_is_correct:
                pure_llm_correct += 1

            no_verify_result = results_by_config.get("ablation_no_verify", {})
            no_verify_is_correct = no_verify_result.get('is_correct', False)
            if no_verify_is_correct:
                no_verify_correct += 1

            no_aggregator_result = results_by_config.get("ablation_no_aggregator", {})
            no_aggregator_is_correct = no_aggregator_result.get('is_correct', False)
            if no_aggregator_is_correct:
                no_aggregator_correct += 1

            case_type = determine_case_type(full_is_correct, no_history_is_correct)

            history_quality_label = analyze_history_quality_v2(full_result.get('agent_details', '[]'))

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
                'full_prediction': full_result.get('prediction', ''),
                'full_predicted_gloss': full_result.get('predicted_gloss', ''),
                'full_reason': full_result.get('reason', ''),
                'full_supporting_sentences': full_result.get('supporting_sentences', ''),
                'nohistory_prediction': no_history_result.get('prediction', ''),
                'nohistory_predicted_gloss': no_history_result.get('predicted_gloss', ''),
                'nohistory_reason': no_history_result.get('reason', ''),
                'purellm_prediction': pure_llm_result.get('prediction', ''),
                'purellm_predicted_gloss': pure_llm_result.get('predicted_gloss', ''),
                'purellm_reason': pure_llm_result.get('reason', ''),
                'no_verify_correct': no_verify_is_correct,
                'no_verify_prediction': no_verify_result.get('prediction', ''),
                'no_aggregator_correct': no_aggregator_is_correct,
                'no_aggregator_prediction': no_aggregator_result.get('prediction', ''),
                'agent_raw_count': full_result.get('agent_raw_count', 0),
                'agent_valid_count': full_result.get('agent_valid_count', 0),
                'agent_filter_rate': full_result.get('agent_filter_rate', 1.0),
                'agent_support_pattern': full_result.get('agent_support_pattern', ''),
                'agent_all_no_evidence': full_result.get('agent_all_no_evidence', False),
                'agent_details': full_result.get('agent_details', ''),
                'aggregator_degraded': full_result.get('aggregator_degraded', False),
                'aggregator_step1': full_result.get('aggregator_step1', ''),
                'aggregator_step2': full_result.get('aggregator_step2', '')
            }

            all_results.append(merged_result)

        full_accuracy = full_correct / len(sampled_senses) if sampled_senses else 0.0
        no_history_accuracy = no_history_correct / len(sampled_senses) if sampled_senses else 0.0
        pure_llm_accuracy = pure_llm_correct / len(sampled_senses) if sampled_senses else 0.0

        print(f"\nRound {round_num + 1} results:")
        print(f"  Full configuration: {full_correct}/{len(sampled_senses)} = {full_accuracy:.3f}")
        print(f"  Ablation_no_history: {no_history_correct}/{len(sampled_senses)} = {no_history_accuracy:.3f}")
        print(f"  Pure LLM: {pure_llm_correct}/{len(sampled_senses)} = {pure_llm_accuracy:.3f}")

        round_results.append({
            'round': round_num + 1,
            'full_accuracy': full_accuracy,
            'no_history_accuracy': no_history_accuracy,
            'pure_llm_accuracy': pure_llm_accuracy,
            'no_verify_accuracy': no_verify_correct / len(sampled_senses) if sampled_senses else 0,
            'no_aggregator_accuracy': no_aggregator_correct / len(sampled_senses) if sampled_senses else 0,
            'difference': no_history_accuracy - full_accuracy
        })

    round_df = pd.DataFrame(round_results)
    print(f"\n{'=' * 90}")
    print("Round Results Summary:")
    print(f"{'=' * 90}")
    print(round_df)

    print(f"\n{'=' * 90}")
    print("Paired t-test Results:")
    print(f"{'=' * 90}")

    full_scores = round_df['full_accuracy'].tolist()
    no_history_scores = round_df['no_history_accuracy'].tolist()
    differences = round_df['difference'].tolist()

    t_stat, p_value = stats.ttest_rel(no_history_scores, full_scores)

    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)

    if n > 1 and std_diff > 0:
        ci = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))
    else:
        ci = (np.nan, np.nan)

    print(f"Mean difference (no_history - full): {mean_diff:.4f}")
    print(f"Standard deviation: {std_diff:.4f}")
    print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

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

    print(f"\n{'=' * 90}")
    print("Exporting detailed results...")
    print(f"{'=' * 90}")

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        detail_df = pd.DataFrame(all_results)

        columns_order = [
            'round', 'sample_id', 'word', 'context', 'gold_wsid', 'gold_gloss',
            'candidate_count', 'mean_pairwise_similarity', 'max_pairwise_similarity', 'top2_similarity',
            'topk_candidate_count', 'topk_mean_pairwise_similarity', 'topk_max_pairwise_similarity', 'topk_top2_similarity',
            'gold_in_topk_union', 'onlyllm_correct', 'nohistory_correct', 'full_correct',
            'case_type', 'history_quality_label',
            'full_prediction', 'full_predicted_gloss', 'full_reason', 'full_supporting_sentences',
            'nohistory_prediction', 'nohistory_predicted_gloss', 'nohistory_reason',
            'purellm_prediction', 'purellm_predicted_gloss', 'purellm_reason',
            'no_verify_correct', 'no_aggregator_correct',
            'no_verify_prediction', 'no_aggregator_prediction',
            'agent_raw_count', 'agent_valid_count', 'agent_filter_rate',
            'agent_support_pattern', 'agent_all_no_evidence', 'agent_details',
            'aggregator_degraded', 'aggregator_step1', 'aggregator_step2'
        ]
        detail_df = detail_df[columns_order]

        excel_file = f'local_comparison_full_vs_nohistory_nopseudo_detailed_{timestamp}.xlsx'
        detail_df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Successfully exported detailed results to {excel_file}")

        csv_file = f'local_comparison_full_vs_nohistory_nopseudo_detailed_{timestamp}.csv'
        detail_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ Successfully exported detailed results to {csv_file}")

        round_excel_file = f'local_comparison_full_vs_nohistory_nopseudo_rounds_{timestamp}.xlsx'
        round_df.to_excel(round_excel_file, index=False, engine='openpyxl')
        print(f"✅ Successfully exported round results to {round_excel_file}")

    except Exception as e:
        print(f"⚠️  Failed to export results: {e}")

    return round_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare full vs ablation_no_history (NO PSEUDO - direct gloss search)')
    parser.add_argument('--rounds', type=int, default=2, help='Number of rounds (default: 2)')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples per round (default: 20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_comparison_test(num_rounds=args.rounds, num_samples_per_round=args.samples)
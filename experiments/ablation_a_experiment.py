# ablation_a_experiment.py
"""
消融实验A：测试LLM筛选步骤的有效性（本地版本）
- 检索到的历史句子不做LLM筛选
- 全部句子直接传给聚合器
- 按句子数量填充contextual_fit/confidence/reason
"""

import json
import sys
import io
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = str(BASE_DIR / "bm25_index_zh")
DATA1_PATH = str(BASE_DIR / "dataprocess" / "data" / "data1.xlsx")
DATA2_PATH = str(BASE_DIR / "dataprocess" / "data" / "data2.xlsx")


def import_modules():
    global load_test_data, calculate_sense_similarity_metrics, generate_topk_union
    global aggregator, analyze_history_quality_v2
    global get_supporting_sentences_without_pseudo_bm25
    global call_llm, cached_call_llm
    
    from main_experiment import (
        load_test_data, calculate_sense_similarity_metrics,
        generate_topk_union, aggregator, analyze_history_quality_v2
    )
    from pseudo_retrieval import get_supporting_sentences_with_pseudo_bm25
    from llm import call_llm, cached_call_llm


def sense_agent_no_verify(text, word, sense_gloss, sense_wsid, raw_sentences):
    """
    消融实验A的义项智能体：不做LLM筛选，直接返回所有含目标词的句子
    按句子数量填充元信息，不硬编码unknown
    """
    # 过滤含目标词的句子（不截断）
    candidates = [
        s for s in raw_sentences
        if word in (s["sentence"] if isinstance(s, dict) else str(s))
    ]
    
    # 不截断，全部传给聚合器（去掉[:3]）
    valid_examples = [
        s["sentence"] if isinstance(s, dict) else str(s)
        for s in candidates
    ]
    
    # 按数量填充元信息，不能硬编码unknown
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


def process_ablation_a(text, word, gold_wsid, candidate_senses):
    """运行消融实验A：不做LLM筛选的full配置"""
    # Top-k union (使用k=2)
    top2_senses, _, _ = generate_topk_union(text, candidate_senses, k=2)
    topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
    gold_in_topk_union = gold_wsid in topk_union_wsids

    # 义项智能体处理（不做LLM筛选）
    agent_results = []
    agent_raw_counts = []

    for sense in top2_senses:
        sense_wsid = str(sense["wsid"])
        sense_gloss = sense.get("newgloss", "")

        try:
            raw_sentences = get_supporting_sentences_with_pseudo_bm25(
                sense_gloss=sense_gloss,
                target_word=word,
                index_dir=INDEX_DIR,
                wsid=sense_wsid,
                bm25_top_k=200,
                final_top_k=10,
                threshold=0.5
            )
        except Exception as e:
            print(f"检索失败: {e}")
            raw_sentences = []

        # 消融A：不做LLM筛选
        agent_output = sense_agent_no_verify(
            text=text,
            word=word,
            sense_gloss=sense_gloss,
            sense_wsid=sense_wsid,
            raw_sentences=raw_sentences
        )
        agent_results.append(agent_output)
        raw_count = len([s for s in raw_sentences if word in (s["sentence"] if isinstance(s, dict) else str(s))])
        agent_raw_counts.append(raw_count)

    # 聚合器决策
    if agent_results:
        prediction, reason, aggregator_step1, aggregator_step2 = aggregator(
            text=text,
            target_word=word,
            agent_results=agent_results,
            call_llm=call_llm
        )

        predicted_gloss = ""
        for sense in candidate_senses:
            if str(sense["wsid"]) == prediction:
                predicted_gloss = sense.get("newgloss", "")
                break

        is_correct = prediction == gold_wsid
    else:
        prediction = str(candidate_senses[0]["wsid"]) if candidate_senses else ""
        predicted_gloss = ""
        is_correct = False
        reason = "无候选义项"
        aggregator_step1 = "N/A"
        aggregator_step2 = "N/A"

    # 计算agent统计
    agent_details = json.dumps([{
        "wsid": a["wsid"],
        "gloss": a["gloss"],
        "raw_count": agent_raw_counts[i] if i < len(agent_raw_counts) else 0,
        "valid_count": len(a["valid_examples"]),
        "valid_examples": a["valid_examples"],
        "support": a["support"],
        "confidence": a["confidence"],
        "evidence_status": a["evidence_status"],
        "contextual_fit": a["contextual_fit"],
        "reason": a["reason"]
    } for i, a in enumerate(agent_results)], ensure_ascii=False)

    history_quality = analyze_history_quality_v2(agent_details)

    return {
        'prediction': prediction,
        'predicted_gloss': predicted_gloss,
        'is_correct': is_correct,
        'reason': reason,
        'gold_in_topk_union': gold_in_topk_union,
        'topk_union_wsids': ','.join(topk_union_wsids),
        'agent_raw_count': sum(agent_raw_counts),
        'agent_valid_count': sum(len(a["valid_examples"]) for a in agent_results),
        'agent_filter_rate': 0.0,  # 消融A不做筛选，过滤率为0
        'agent_all_no_evidence': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'agent_details': agent_details,
        'aggregator_degraded': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'aggregator_step1': aggregator_step1,
        'aggregator_step2': aggregator_step2,
        'history_quality_label': history_quality
    }


def run_ablation_a_on_single_file(data_path, source_name):
    """对单个文件运行消融实验A"""
    print(f"\n{'='*70}")
    print(f"消融实验A - 处理 {source_name}")
    print(f"{'='*70}")
    
    import_modules()
    
    contexts, all_senses, wsid_to_context, wsid_to_sense, all_word_to_senses, _ = load_test_data()
    
    df = pd.read_excel(data_path)
    print(f"样本数: {len(df)}")
    
    all_results = []
    correct_count = 0
    
    for idx, row in df.iterrows():
        word = row['word']
        text = row['context']
        gold_wsid = str(row['gold_wsid'])
        gold_gloss = row['gold_gloss']
        
        candidate_senses = all_word_to_senses.get(word, [])
        
        if not candidate_senses:
            print(f"警告：未找到词 '{word}' 的候选义项")
            continue
        
        print(f"\n[{source_name}] 处理样本 {idx+1}/{len(df)}: {word}")
        
        result = process_ablation_a(text, word, gold_wsid, candidate_senses)
        
        correct = result['is_correct']
        if correct:
            correct_count += 1
        
        sim_metrics = calculate_sense_similarity_metrics(candidate_senses)
        
        all_results.append({
            'round': row['round'],
            'sample_id': row['sample_id'],
            'word': word,
            'context': text,
            'gold_wsid': gold_wsid,
            'gold_gloss': gold_gloss,
            'candidate_count': len(candidate_senses),
            'mean_pairwise_similarity': sim_metrics['mean_pairwise_similarity'],
            'max_pairwise_similarity': sim_metrics['max_pairwise_similarity'],
            'ablation_a_correct': correct,
            'ablation_a_prediction': result['prediction'],
            'ablation_a_predicted_gloss': result['predicted_gloss'],
            'ablation_a_reason': result['reason'],
            'history_quality_label': result['history_quality_label'],
            'agent_raw_count': result['agent_raw_count'],
            'agent_valid_count': result['agent_valid_count'],
            'agent_filter_rate': result['agent_filter_rate'],
            'agent_all_no_evidence': result['agent_all_no_evidence'],
            'aggregator_degraded': result['aggregator_degraded'],
            'aggregator_step1': result['aggregator_step1'],
            'aggregator_step2': result['aggregator_step2'],
            'agent_details': result['agent_details']
        })
        
        print(f"  结果: {'✓' if correct else '✗'}")
    
    accuracy = correct_count / len(all_results)
    print(f"\n{source_name} 消融A准确率: {correct_count}/{len(all_results)} = {accuracy:.4f}")
    
    return all_results, accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验A：测试LLM筛选步骤的有效性')
    parser.add_argument('--target', type=str, choices=['data1', 'data2', 'both'], default='both',
                        help='运行目标: data1, data2, 或 both')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {}
    
    if args.target in ['data1', 'both']:
        all_results, acc = run_ablation_a_on_single_file(DATA1_PATH, 'data1')
        results['data1'] = (all_results, acc)
        
        df = pd.DataFrame(all_results)
        output_file = f'ablation_a_data1_{timestamp}.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ data1 消融A结果已导出到 {output_file}")
    
    if args.target in ['data2', 'both']:
        all_results, acc = run_ablation_a_on_single_file(DATA2_PATH, 'data2')
        results['data2'] = (all_results, acc)
        
        df = pd.DataFrame(all_results)
        output_file = f'ablation_a_data2_{timestamp}.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ data2 消融A结果已导出到 {output_file}")
    
    print(f"\n{'='*70}")
    print("消融实验A运行完成")
    print(f"{'='*70}")
    for name, (_, acc) in results.items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
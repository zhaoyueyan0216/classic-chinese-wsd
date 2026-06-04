# ablation_nohistory_experiment.py
"""
消融实验 No-History：测试历史证据的有效性
- 保留 TopK 候选筛选步骤
- 不检索任何历史例句
- 聚合器仅凭义项定义和目标句语境判断（degraded 模式）
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
DATA1_PATH = str(BASE_DIR / "dataprocess" / "data" / "data1.xlsx")
DATA2_PATH = str(BASE_DIR / "dataprocess" / "data" / "data2.xlsx")


def import_modules():
    global load_test_data, calculate_sense_similarity_metrics, generate_topk_union
    global aggregator, analyze_history_quality_v2
    global call_llm

    from main_experiment import (
        load_test_data, calculate_sense_similarity_metrics,
        generate_topk_union, aggregator, analyze_history_quality_v2
    )
    from llm import call_llm


def process_ablation_nohistory(text, word, gold_wsid, candidate_senses):
    """运行消融 No-History：TopK 筛选后不检索历史证据，聚合器凭义项定义判断"""
    top2_senses, _, _ = generate_topk_union(text, candidate_senses, k=2)
    topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
    gold_in_topk_union = gold_wsid in topk_union_wsids

    # 不检索，直接构造空证据
    agent_results = [
        {
            "wsid": str(sense["wsid"]),
            "gloss": sense.get("newgloss", ""),
            "valid_examples": [],
            "evidence_status": "unknown",
            "contextual_fit": "unknown",
            "support": None,
            "confidence": "unknown",
            "reason": "未启用历史证据检索"
        }
        for sense in top2_senses
    ]

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

    agent_details = json.dumps([{
        "wsid": a["wsid"],
        "gloss": a["gloss"],
        "valid_count": 0,
        "valid_examples": [],
        "confidence": a["confidence"],
        "reason": a["reason"]
    } for a in agent_results], ensure_ascii=False)

    return {
        'prediction': prediction,
        'predicted_gloss': predicted_gloss,
        'is_correct': is_correct,
        'reason': reason,
        'gold_in_topk_union': gold_in_topk_union,
        'topk_union_wsids': ','.join(topk_union_wsids),
        'aggregator_step1': aggregator_step1,
        'aggregator_step2': aggregator_step2,
        'agent_details': agent_details
    }


def run_ablation_nohistory_on_single_file(data_path, source_name):
    """对单个文件运行消融 No-History"""
    print(f"\n{'='*70}")
    print(f"消融实验 No-History - 处理 {source_name}")
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

        result = process_ablation_nohistory(text, word, gold_wsid, candidate_senses)

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
            'ablation_nohistory_correct': correct,
            'ablation_nohistory_prediction': result['prediction'],
            'ablation_nohistory_predicted_gloss': result['predicted_gloss'],
            'ablation_nohistory_reason': result['reason'],
            'gold_in_topk_union': result['gold_in_topk_union'],
            'aggregator_step1': result['aggregator_step1'],
            'aggregator_step2': result['aggregator_step2'],
            'agent_details': result['agent_details']
        })

        print(f"  结果: {'✓' if correct else '✗'}")

    accuracy = correct_count / len(all_results) if all_results else 0.0
    print(f"\n{source_name} 消融 No-History 准确率: {correct_count}/{len(all_results)} = {accuracy:.4f}")

    return all_results, accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验 No-History：测试历史证据的有效性')
    parser.add_argument('--target', type=str, choices=['data1', 'data2', 'both'], default='both',
                        help='运行目标: data1, data2, 或 both')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {}

    if args.target in ['data1', 'both']:
        all_results, acc = run_ablation_nohistory_on_single_file(DATA1_PATH, 'data1')
        results['data1'] = (all_results, acc)
        df = pd.DataFrame(all_results)
        output_file = f'ablation_nohistory_data1_{timestamp}.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ data1 消融 No-History 结果已导出到 {output_file}")

    if args.target in ['data2', 'both']:
        all_results, acc = run_ablation_nohistory_on_single_file(DATA2_PATH, 'data2')
        results['data2'] = (all_results, acc)
        df = pd.DataFrame(all_results)
        output_file = f'ablation_nohistory_data2_{timestamp}.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ data2 消融 No-History 结果已导出到 {output_file}")

    print(f"\n{'='*70}")
    print("消融实验 No-History 运行完成")
    print(f"{'='*70}")
    for name, (_, acc) in results.items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()

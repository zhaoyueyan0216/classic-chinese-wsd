# no_pseudo_only_experiment.py
"""
只运行 no_pseudo（gloss直接检索）的 full 配置
不运行其他配置（ablation_no_history, pure_llm, ablation_no_verify, ablation_no_aggregator）
"""

import json
import sys
import io
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

from main_experiment import (
    load_test_data, calculate_sense_similarity_metrics,
    generate_topk_union, sense_agent, aggregator, analyze_history_quality_v2
)
from core.pseudo_retrieval import get_supporting_sentences_without_pseudo_bm25
from core.llm import call_llm, cached_call_llm
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

INDEX_DIR = str(Path(__file__).parent / "bm25_index_zh")


def process_nopseudo_full(text, candidate_senses, gold_wsid):
    """
    只运行 no_pseudo full 配置
    """
    word = candidate_senses[0]['word'] if candidate_senses else "unknown"

    # Top-k union
    top2_senses, _, _ = generate_topk_union(text, candidate_senses, k=2)
    topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
    gold_in_topk_union = gold_wsid in topk_union_wsids

    # 义项智能体处理
    agent_results = []
    agent_raw_counts = []

    for sense in top2_senses:
        sense_wsid = str(sense["wsid"])
        sense_gloss = sense.get("newgloss", "")

        # 使用 no_pseudo 检索
        try:
            raw_sentences = get_supporting_sentences_without_pseudo_bm25(
                sense_gloss=sense_gloss,
                target_word=word,
                index_dir=INDEX_DIR,
                wsid=sense_wsid,
                bm25_top_k=200,
                final_top_k=10,
                threshold=0.5
            )
        except Exception:
            raw_sentences = []

        # 义项智能体验证
        agent_output = sense_agent(
            text=text,
            target_word=word,
            sense_gloss=sense_gloss,
            sense_wsid=sense_wsid,
            raw_sentences=raw_sentences,
            cached_call_llm=cached_call_llm
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

        # 获取预测的gloss
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
        'agent_filter_rate': round(1 - sum(len(a["valid_examples"]) for a in agent_results) / sum(agent_raw_counts), 2) if sum(agent_raw_counts) > 0 else 1.0,
        'agent_all_no_evidence': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'agent_details': agent_details,
        'aggregator_degraded': all(len(a["valid_examples"]) == 0 for a in agent_results),
        'aggregator_step1': aggregator_step1,
        'aggregator_step2': aggregator_step2,
        'history_quality_label': history_quality
    }


def run_nopseudo_only_experiment(num_rounds=2, num_samples=20):
    print("=" * 70)
    print("NO_PSEUDO实验：只运行 gloss直接检索 full 配置")
    print(f"设置：{num_rounds}轮 × {num_samples}样本")
    print("=" * 70)

    contexts, all_senses, wsid_to_context, wsid_to_sense, all_word_to_senses, _ = load_test_data()

    # 过滤多义词
    filtered = [ctx for ctx in contexts
                if wsid_to_sense.get(str(ctx["wsid"])) and
                len(all_word_to_senses.get(wsid_to_sense[str(ctx["wsid"])]["word"], [])) > 1]
    print(f"可用样本数: {len(filtered)}")

    all_results = []
    round_results = []

    for round_num in range(num_rounds):
        print(f"\n轮次 {round_num+1}/{num_rounds}")
        sampled = random.sample(filtered, min(num_samples, len(filtered)))

        correct_count = 0

        for idx, ctx in enumerate(sampled):
            wsid = str(ctx["wsid"])
            sense = wsid_to_sense[wsid]
            word = sense["word"]
            text = ctx["txt"]
            gold_wsid = wsid
            gold_gloss = sense.get("newgloss", "")
            candidate_senses = all_word_to_senses[word]

            print(f"\n处理样本 {idx+1}/{len(sampled)}: {word}")

            # 只运行 no_pseudo full 配置
            result = process_nopseudo_full(text, candidate_senses, gold_wsid)

            correct = result['is_correct']
            if correct:
                correct_count += 1

            # 计算相似度指标
            sim_metrics = calculate_sense_similarity_metrics(candidate_senses)

            all_results.append({
                'round': round_num+1,
                'sample_id': idx+1,
                'word': word,
                'context': text,
                'gold_wsid': gold_wsid,
                'gold_gloss': gold_gloss,
                'candidate_count': len(candidate_senses),
                'mean_pairwise_similarity': sim_metrics['mean_pairwise_similarity'],
                'max_pairwise_similarity': sim_metrics['max_pairwise_similarity'],
                'topk_candidate_count': len([s for s in candidate_senses if str(s["wsid"]) in result['topk_union_wsids'].split(',')]),
                'gold_in_topk_union': 'yes' if result['gold_in_topk_union'] else 'no',
                'nopseudo_correct': correct,
                'nopseudo_prediction': result['prediction'],
                'nopseudo_predicted_gloss': result['predicted_gloss'],
                'nopseudo_reason': result['reason'],
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

        # 计算本轮准确率
        accuracy = correct_count / len(sampled)
        print(f"\n轮次 {round_num+1} 结果:")
        print(f"  no_pseudo full: {correct_count}/{len(sampled)} = {accuracy:.3f}")

        round_results.append({
            'round': round_num+1,
            'nopseudo_accuracy': accuracy
        })

    # 统计分析
    print("\n" + "=" * 70)
    print("统计分析结果")
    print("=" * 70)

    round_df = pd.DataFrame(round_results)
    print("\n每轮结果:")
    print(round_df)

    print(f"\nno_pseudo平均准确率: {np.mean([r['nopseudo_accuracy'] for r in round_results]):.4f}")
    print(f"no_pseudo标准差: {np.std([r['nopseudo_accuracy'] for r in round_results], ddof=1):.4f}")

    # 导出结果
    print("\n" + "=" * 70)
    print("导出结果")
    print("=" * 70)

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 详细结果
        detail_df = pd.DataFrame(all_results)
        detail_excel = f'nopseudo_only_detailed_{timestamp}.xlsx'
        detail_df.to_excel(detail_excel, index=False, engine='openpyxl')
        print(f"✅ 详细结果导出到 {detail_excel}")

        # 轮次结果
        round_excel = f'nopseudo_only_rounds_{timestamp}.xlsx'
        round_df.to_excel(round_excel, index=False, engine='openpyxl')
        print(f"✅ 轮次结果导出到 {round_excel}")

    except Exception as e:
        print(f"⚠️  导出失败: {e}")

    return round_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NO_PSEUDO实验：只运行 gloss直接检索 full 配置')
    parser.add_argument('--rounds', type=int, default=2, help='轮次数（默认：2）')
    parser.add_argument('--samples', type=int, default=20, help='每轮样本数（默认：20）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_nopseudo_only_experiment(num_rounds=args.rounds, num_samples=args.samples)
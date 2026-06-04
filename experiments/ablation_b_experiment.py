# ablation_b_experiment.py
"""
消融实验B：测试聚合器跨义项综合推理的有效性（本地版本）
- 使用完整的义项智能体（带LLM筛选）
- 不使用聚合器，直接拼接所有义项智能体的输出
- 让普通LLM做最终判断，不做跨义项比较推理
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
    global sense_agent, analyze_history_quality_v2
    global get_supporting_sentences_with_pseudo_bm25
    global call_llm, cached_call_llm

    from main_experiment import (
        load_test_data, calculate_sense_similarity_metrics,
        generate_topk_union, sense_agent, analyze_history_quality_v2
    )
    from pseudo_retrieval import get_supporting_sentences_with_pseudo_bm25
    from llm import call_llm, cached_call_llm


def vote_decision(text, target_word, agent_results, call_llm):
    """
    消融B：不用聚合器，直接拼接所有义项智能体的输出
    让LLM基于拼接结果做最终判断，不做跨义项比较推理
    """
    # 拼接所有义项智能体的报告（包含wsid）
    concatenated = ""
    for i, agent in enumerate(agent_results):
        concatenated += f"义项{i+1}（wsid={agent['wsid']}）：{agent['gloss']}\n"
        concatenated += f"判断：{agent['reason']}\n"
        if agent.get('valid_examples'):
            concatenated += f"例句：{agent['valid_examples'][0]}\n"
        concatenated += "\n"

    # 用一个普通LLM直接判断，不做跨义项比较
    prompt = f"""你是古汉语词义专家。

目标句：{text}
目标词：「{target_word}」

以下是各候选义项的独立分析结果：
{concatenated}
请直接选出最符合目标句的义项，输出该义项的wsid。

输出严格JSON：
{{"prediction": "wsid数字", "reason": "判断理由"}}"""

    response = call_llm(prompt)

    try:
        result = json.loads(response)
        prediction = str(result["prediction"])
        reason = result.get("reason", "")
    except (json.JSONDecodeError, KeyError) as e:
        # 解析失败时选择第一个义项
        prediction = str(agent_results[0]["wsid"]) if agent_results else ""
        reason = f"解析失败: {e}"

    return (
        prediction,
        reason,
        "拼接模式：无聚合器",
        "N/A"
    )


def process_ablation_b(text, word, gold_wsid, candidate_senses):
    """运行消融实验B：不用聚合器，直接拼接"""
    # Top-k union (使用k=2)
    top2_senses, _, _ = generate_topk_union(text, candidate_senses, k=2)
    topk_union_wsids = [str(s["wsid"]) for s in top2_senses]
    gold_in_topk_union = gold_wsid in topk_union_wsids

    # 义项智能体处理（使用完整的sense_agent，带LLM筛选）
    agent_results = []
    agent_raw_counts = []
    agent_valid_counts = []

    for sense in top2_senses:
        sense_wsid = str(sense["wsid"])
        sense_gloss = sense.get("newgloss", "")

        # 使用 pseudo 检索（获取句子）
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

        # 完整的义项智能体（带LLM筛选）
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
        agent_valid_counts.append(len(agent_output.get("valid_examples", [])))

    # 消融B：使用vote_decision替代aggregator
    if agent_results:
        prediction, reason, aggregator_mode, aggregator_step2 = vote_decision(
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
        aggregator_mode = "N/A"
        aggregator_step2 = "N/A"

    # 计算agent统计
    agent_details = json.dumps([{
        "wsid": a["wsid"],
        "gloss": a["gloss"],
        "raw_count": agent_raw_counts[i] if i < len(agent_raw_counts) else 0,
        "valid_count": agent_valid_counts[i] if i < len(agent_valid_counts) else 0,
        "valid_examples": a.get("valid_examples", []),
        "support": a.get("support"),
        "confidence": a.get("confidence"),
        "evidence_status": a.get("evidence_status"),
        "contextual_fit": a.get("contextual_fit"),
        "reason": a.get("reason")
    } for i, a in enumerate(agent_results)], ensure_ascii=False)

    history_quality = analyze_history_quality_v2(agent_details)

    # 消融B：过滤率不为0，因为使用了完整的sense_agent
    total_raw = sum(agent_raw_counts)
    total_valid = sum(agent_valid_counts)
    filter_rate = (total_raw - total_valid) / total_raw if total_raw > 0 else 0.0

    return {
        'prediction': prediction,
        'predicted_gloss': predicted_gloss,
        'is_correct': is_correct,
        'reason': reason,
        'gold_in_topk_union': gold_in_topk_union,
        'topk_union_wsids': ','.join(topk_union_wsids),
        'agent_raw_count': total_raw,
        'agent_valid_count': total_valid,
        'agent_filter_rate': filter_rate,
        'agent_all_no_evidence': all(len(a.get("valid_examples", [])) == 0 for a in agent_results),
        'agent_details': agent_details,
        'aggregator_mode': aggregator_mode,
        'aggregator_step2': aggregator_step2,
        'history_quality_label': history_quality
    }


def run_ablation_b_on_single_file(data_path, source_name):
    """对单个文件运行消融实验B"""
    print(f"\n{'='*70}")
    print(f"消融实验B - 处理 {source_name}")
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

        result = process_ablation_b(text, word, gold_wsid, candidate_senses)

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
            'ablation_b_correct': correct,
            'ablation_b_prediction': result['prediction'],
            'ablation_b_predicted_gloss': result['predicted_gloss'],
            'ablation_b_reason': result['reason'],
            'history_quality_label': result['history_quality_label'],
            'agent_raw_count': result['agent_raw_count'],
            'agent_valid_count': result['agent_valid_count'],
            'agent_filter_rate': result['agent_filter_rate'],
            'agent_all_no_evidence': result['agent_all_no_evidence'],
            'aggregator_mode': result['aggregator_mode'],
            'aggregator_step2': result['aggregator_step2'],
            'agent_details': result['agent_details']
        })

        print(f"  结果: {'✓' if correct else '✗'}")

    accuracy = correct_count / len(all_results)
    print(f"\n{source_name} 消融B准确率: {correct_count}/{len(all_results)} = {accuracy:.4f}")

    return all_results, accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验B：测试聚合器跨义项综合推理的有效性')
    parser.add_argument('--target', type=str, choices=['data1', 'data2', 'both'], default='both',
                        help='运行目标: data1, data2, 或 both')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {}

    if args.target in ['data1', 'both']:
        all_results, acc = run_ablation_b_on_single_file(DATA1_PATH, 'data1')
        results['data1'] = (all_results, acc)

        df = pd.DataFrame(all_results)
        output_file = f'ablation_b_data1_{timestamp}.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ data1 消融B结果已导出到 {output_file}")

    if args.target in ['data2', 'both']:
        all_results, acc = run_ablation_b_on_single_file(DATA2_PATH, 'data2')
        results['data2'] = (all_results, acc)

        df = pd.DataFrame(all_results)
        output_file = f'ablation_b_data2_{timestamp}.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ data2 消融B结果已导出到 {output_file}")

    print(f"\n{'='*70}")
    print("消融实验B运行完成")
    print(f"{'='*70}")
    for name, (_, acc) in results.items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
# naive_rag_experiment.py
"""
NaiveRAG实验：只运行朴素RAG
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

from main_experiment import load_test_data, calculate_sense_similarity_metrics
from core.pseudo_retrieval import get_supporting_sentences_without_pseudo_bm25
from core.llm import call_llm
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

INDEX_DIR = str(Path(__file__).parent / "bm25_index_zh")


def process_naive_rag(text, word, candidate_senses):
    """
    NaiveRAG：用目标句直接检索，结果直接喂给LLM
    不做TopK筛选，不做义项感知检索，不做验证，不做聚合
    """
    # 用目标句直接检索
    try:
        raw_sentences = get_supporting_sentences_without_pseudo_bm25(
            sense_gloss=text,   # 目标句作为query
            target_word=word,
            index_dir=INDEX_DIR,
            wsid="",
            bm25_top_k=200,
            final_top_k=15,
            threshold=0.5
        )
    except Exception as e:
        print(f"[NaiveRAG] 检索失败: {e}")
        raw_sentences = []

    # 过滤含目标词的句子
    valid_sents = [
        s["sentence"] if isinstance(s, dict) else str(s)
        for s in raw_sentences
        if word in (s["sentence"] if isinstance(s, dict) else str(s))
    ][:10]

    # 构建候选义项block（用所有候选义项，不做TopK）
    sense_block = "\n".join(
        f"wsid={s['wsid']}: {s.get('newgloss','')}"
        for s in candidate_senses
    )

    # 构建检索结果block
    if valid_sents:
        sents_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(valid_sents)
        )
        evidence_part = f"\n[检索到的历史例句]\n{sents_block}\n"
    else:
        evidence_part = "\n[未检索到含目标词的历史例句]\n"

    # 直接判断
    prompt = f"""你是古汉语词义消歧专家。

目标句：{text}
目标词：「{word}」
{evidence_part}
候选义项：
{sense_block}

请选出目标句中「{word}」最符合的义项。

输出严格JSON：
{{"prediction": "wsid", "reason": "判断理由"}}"""

    response = call_llm(prompt)
    response = response.strip()
    for prefix in ["```json", "```"]:
        if response.startswith(prefix):
            response = response[len(prefix):]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        res = json.loads(response)
        return str(res.get("prediction", "")), res.get("reason", ""), len(valid_sents)
    except Exception as e:
        return str(candidate_senses[0]["wsid"]) if candidate_senses else "", f"解析失败:{e}", 0


def run_naive_rag_experiment(num_rounds=5, num_samples=100, random_seed=42):
    """
    运行NaiveRAG实验
    :param num_rounds: 轮次数
    :param num_samples: 每轮样本数
    :param random_seed: 随机种子，默认42
    """
    print("=" * 70)
    print("NaiveRAG实验：只运行朴素RAG")
    print(f"设置：{num_rounds}轮 × {num_samples}样本")
    print(f"随机种子: {random_seed}")
    print("=" * 70)

    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    contexts, all_senses, wsid_to_context, wsid_to_sense, \
        all_word_to_senses, _ = load_test_data()

    filtered = [ctx for ctx in contexts
                if wsid_to_sense.get(str(ctx["wsid"])) and
                len(all_word_to_senses.get(
                    wsid_to_sense[str(ctx["wsid"])]["word"], [])) > 1]
    print(f"可用样本数: {len(filtered)}")

    all_results = []
    round_results = []

    for round_num in range(num_rounds):
        print(f"\n轮次 {round_num+1}/{num_rounds}")
        sampled = random.sample(filtered, min(num_samples, len(filtered)))

        naive_correct = 0
        total_retrieved = 0

        for idx, ctx in enumerate(sampled):
            wsid = str(ctx["wsid"])
            sense = wsid_to_sense[wsid]
            word = sense["word"]
            text = ctx["txt"]
            gold_wsid = wsid
            gold_gloss = sense.get("newgloss", "")
            candidate_senses = all_word_to_senses[word]

            # 只跑 NaiveRAG
            pred, reason, retrieved_count = process_naive_rag(text, word, candidate_senses)
            naive_ok = (pred == gold_wsid)

            if naive_ok:
                naive_correct += 1
            total_retrieved += retrieved_count

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
                'naive_rag_correct': naive_ok,
                'naive_rag_prediction': pred,
                'naive_rag_reason': reason,
                'retrieved_sentence_count': retrieved_count
            })

            print(f"  样本{idx+1}: NaiveRAG={'✓' if naive_ok else '✗'} (检索到{retrieved_count}条)")

        # 计算本轮准确率
        naive_acc = naive_correct / len(sampled)
        avg_retrieved = total_retrieved / len(sampled)

        print(f"\n轮次 {round_num+1} 结果:")
        print(f"  NaiveRAG准确率: {naive_correct}/{len(sampled)} = {naive_acc:.3f}")
        print(f"  平均检索句子数: {avg_retrieved:.2f}")

        round_results.append({
            'round': round_num+1,
            'naive_rag_accuracy': naive_acc,
            'avg_retrieved_sentences': avg_retrieved
        })

    # 统计分析
    print("\n" + "=" * 70)
    print("统计分析结果")
    print("=" * 70)

    round_df = pd.DataFrame(round_results)
    print("\n每轮结果:")
    print(round_df)

    print(f"\nNaiveRAG平均准确率: {np.mean([r['naive_rag_accuracy'] for r in round_results]):.4f}")
    print(f"NaiveRAG标准差: {np.std([r['naive_rag_accuracy'] for r in round_results], ddof=1):.4f}")

    # 导出结果
    print("\n" + "=" * 70)
    print("导出结果")
    print("=" * 70)

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 详细结果
        detail_df = pd.DataFrame(all_results)
        detail_excel = f'naive_rag_only_detailed_{timestamp}.xlsx'
        detail_df.to_excel(detail_excel, index=False, engine='openpyxl')
        print(f"✅ 详细结果导出到 {detail_excel}")

        # 轮次结果
        round_excel = f'naive_rag_only_rounds_{timestamp}.xlsx'
        round_df.to_excel(round_excel, index=False, engine='openpyxl')
        print(f"✅ 轮次结果导出到 {round_excel}")

    except Exception as e:
        print(f"⚠️  导出失败: {e}")

    return round_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NaiveRAG实验：只运行朴素RAG')
    parser.add_argument('--rounds', type=int, default=5, help='轮次数（默认：5）')
    parser.add_argument('--samples', type=int, default=100, help='每轮样本数（默认：100）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')

    args = parser.parse_args()

    run_naive_rag_experiment(
        num_rounds=args.rounds,
        num_samples=args.samples,
        random_seed=args.seed
    )
# local_add_gold_rank_all.py
"""
计算 gold_wsid 在全部候选义项中的排名位置（本地版本）
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
SENSES_FILE = str(BASE_DIR / "character_senses.json")
DATA1_PATH = str(BASE_DIR / "dataprocess" / "data" / "data1.xlsx")
DATA2_PATH = str(BASE_DIR / "dataprocess" / "data" / "data2.xlsx")


def load_word_to_senses(senses_file):
    """加载义项文件，构建 word_to_senses 字典"""
    with open(senses_file, 'r', encoding='utf-8') as f:
        senses = json.load(f)['RECORDS']

    from collections import defaultdict
    word_to_senses = defaultdict(list)

    for sense in senses:
        word = sense['word']
        wsid = str(sense['wsid'])
        if wsid not in word_to_senses[word]:
            word_to_senses[word].append(wsid)

    print(f"加载了 {len(word_to_senses)} 个词，共 {sum(len(v) for v in word_to_senses.values())} 个义项")
    return word_to_senses


def gold_rank_in_all(row, word_to_senses):
    """计算 gold_wsid 在全部候选义项中的排名（从1开始）"""
    candidates = word_to_senses.get(row['word'], [])
    gold = str(row['gold_wsid'])

    if gold in candidates:
        return candidates.index(gold) + 1
    return None


def process_data(data_path, source_name, word_to_senses):
    """处理单个数据文件"""
    print(f"\n处理 {source_name}...")

    df = pd.read_excel(data_path)
    print(f"原始样本数: {len(df)}")

    df['gold_rank_all'] = df.apply(lambda row: gold_rank_in_all(row, word_to_senses), axis=1)

    valid_count = df['gold_rank_all'].notna().sum()
    avg_rank = df['gold_rank_all'].mean()
    min_rank = df['gold_rank_all'].min()
    max_rank = df['gold_rank_all'].max()

    print(f"有效排名: {valid_count}/{len(df)}")
    print(f"平均排名: {avg_rank:.2f}")
    print(f"最小排名: {min_rank}")
    print(f"最大排名: {max_rank}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{source_name}_with_rank_{timestamp}.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"结果已保存到: {output_file}")

    return df


def main():
    print("="*70)
    print("计算 gold_wsid 在全部候选义项中的排名")
    print("="*70)

    word_to_senses = load_word_to_senses(SENSES_FILE)

    df1 = process_data(DATA1_PATH, 'data1', word_to_senses)
    df2 = process_data(DATA2_PATH, 'data2', word_to_senses)

    print("\n" + "="*70)
    print("处理完成")
    print("="*70)


if __name__ == "__main__":
    main()
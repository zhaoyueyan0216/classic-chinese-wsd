"""LLM-based relevance filtering of the candidates from 01_Context_Selector.py.

Adapted from ACP-RAG_Pipeline/02_Context_Filter.py: uses the project's
DashScope-backed chat_llm() (see acprag_baseline/common.py) instead of a
locally-hosted HF model, but keeps the same 0-3 relevance-scoring prompt and
the "keep candidates scored > 1" rule.
"""

import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from acprag_baseline.common import chat_llm  # noqa: E402

KB_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "acp_qa.json"
INPUT_FILE = PROJECT_ROOT / "acprag_baseline" / "results" / "01_context_selected.json"
OUTPUT_FILE = PROJECT_ROOT / "acprag_baseline" / "results" / "02_context_filtered.json"

RELEVANCE_PROMPT = """文本A：【{query}】
文本B：【{context}】
你是一个严谨、遵守规则的打分专家。请根据上面提供的文本A和文本B，对两者之间的"关联性"进行打分。
打分细则如下：
（1）"关联性"是指文本B的内容是否有助于回答文本A中的问题，主题是否紧密相关。
（2）0分：没有帮助；1分：一点点帮助；2分：基本有帮助；3分：很有帮助。
请根据打分细则给出分数和给分理由。
输出的参考格式如下：
关联性：x分（分数0到3分）
"""


def select_incontext(id_to_record, query, candidates):
    kept = []
    for idx, score in candidates:
        record = id_to_record[idx]
        context = "问题：" + record["Question"] + "答案：" + record["Answer"]
        prompt = RELEVANCE_PROMPT.format(query=query, context=context)

        response = chat_llm(prompt)
        numbers = re.findall(r"\d+", response)

        if numbers and int(numbers[0]) > 1:
            kept.append([idx, score])
    return kept


def main():
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    id_to_record = {r["Id"]: r for r in kb}

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for sample in tqdm(data, desc="Filtering contexts"):
        query = sample["Question"]
        candidates = sample["检索结果"]

        sample_out = dict(sample)
        sample_out["上下文筛选"] = select_incontext(id_to_record, query, candidates)
        results.append(sample_out)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

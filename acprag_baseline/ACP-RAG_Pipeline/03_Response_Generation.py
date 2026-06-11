"""Generate final WSD predictions using the filtered in-context examples.

Adapted from ACP-RAG_Pipeline/03_Response_Generation.py. Unlike a free-text
generation model, this WSD task is evaluated by selecting a sense id (wsid)
from the target character's full candidate-sense inventory
(data/character_senses.json), so the LLM is asked to choose a wsid and
justify it -- the same "wsid=X: gloss" + strict-JSON pattern used by
experiments/main_experiment.py. This makes the output directly comparable
(predicted_wsid vs gold_wsid) to the other methods in data/data_result.xlsx.

Uses core.llm.call_llm (DashScope, response_format=json_object, seed=42 for
determinism) instead of a locally-hosted generation model.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.llm import call_llm  # noqa: E402

KB_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "acp_qa.json"
SENSES_FILE = PROJECT_ROOT / "data" / "character_senses.json"
INPUT_FILE = PROJECT_ROOT / "acprag_baseline" / "results" / "02_context_filtered.json"
OUTPUT_FILE = PROJECT_ROOT / "acprag_baseline" / "results" / "03_response_generated.json"


def load_word_to_senses():
    with open(SENSES_FILE, encoding="utf-8") as f:
        records = json.load(f)["RECORDS"]
    word_to_senses = defaultdict(list)
    for r in records:
        word_to_senses[r["word"]].append({"wsid": str(r["wsid"]), "newgloss": r["newgloss"]})
    return word_to_senses


def get_incontext(id_to_record, candidates):
    if not candidates:
        return ""
    blocks = []
    for i, (context_id, _score) in enumerate(candidates):
        record = id_to_record[context_id]
        blocks.append(f"QA-{i}：\n问题：{record['Question']}\n答案：{record['Answer']}")
    return "<参考资料：古文词义相关问答（可能有助于判断，也可能无关）>\n" + "\n".join(blocks) + "\n\n"


def build_prompt(sample, in_context, candidate_senses):
    senses_block = "\n".join(
        f"wsid={s['wsid']}: {s['newgloss']}" for s in candidate_senses
    )
    return f"""你是古汉语词义消歧专家。

{sample['Question']}

{in_context}候选义项：
{senses_block}

请选出目标词最符合的义项。

输出严格JSON：
{{"prediction": "wsid", "reason": "判断理由"}}"""


def parse_prediction(response, candidate_senses):
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        return str(result["prediction"]), result.get("reason", "")
    except Exception as e:
        fallback = candidate_senses[0]["wsid"] if candidate_senses else ""
        return fallback, f"解析失败: {e}"


def main():
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    id_to_record = {r["Id"]: r for r in kb}

    word_to_senses = load_word_to_senses()

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    n_correct = 0
    for sample in tqdm(data, desc="Generating predictions"):
        candidates = sample["上下文筛选"]
        candidate_senses = word_to_senses.get(sample["word"], [])

        in_context = get_incontext(id_to_record, candidates)
        prompt = build_prompt(sample, in_context, candidate_senses)

        response = call_llm(prompt)
        predicted_wsid, reason = parse_prediction(response, candidate_senses)
        predicted_gloss = next(
            (s["newgloss"] for s in candidate_senses if s["wsid"] == predicted_wsid), ""
        )
        correct = predicted_wsid == str(sample["gold_wsid"])
        n_correct += correct

        sample_out = dict(sample)
        sample_out["Context"] = [
            f"QA-{i}：\n问题：{id_to_record[cid]['Question']}\n答案：{id_to_record[cid]['Answer']}"
            for i, (cid, _score) in enumerate(candidates)
        ]
        sample_out["predicted_wsid"] = predicted_wsid
        sample_out["predicted_gloss"] = predicted_gloss
        sample_out["reason"] = reason
        sample_out["correct"] = correct
        results.append(sample_out)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    accuracy = n_correct / len(results) if results else 0.0
    print(f"Saved {len(results)} results to {OUTPUT_FILE}")
    print(f"Accuracy: {n_correct}/{len(results)} = {accuracy:.4f}")


if __name__ == "__main__":
    main()

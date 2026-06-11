"""Build (query, candidate, label) cross-encoder training pairs from train_data.json.

train_data.json and acprag_baseline/data/acp_qa.json are 1:1 (same order, same
length): row i of train_data.json became record Id=i of acp_qa.json. For each
anchor row i, we construct up to 3 pairs:

  - positive (label=1): anchor's query vs. another knowledge-base record with
    the SAME wsid (same character, same sense, different sentence)
  - hard negative (label=0): anchor's query vs. a record with the SAME
    character but a DIFFERENT wsid (same character, different sense) -- this
    is the case the reranker most needs to learn to reject
  - easy negative (label=0): anchor's query vs. a record for a different
    character entirely

Anchors whose sense has no other context in train_data.json (so no positive
can be formed) are skipped.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_ROOT / "data" / "train_data.json"
KB_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "acp_qa.json"
OUTPUT_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "rank_pairs.jsonl"


def candidate_text(record):
    return f"问题：{record['Question']}答案：{record['Answer']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    with open(TRAIN_FILE, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    assert len(train_data) == len(kb), "train_data.json and acp_qa.json must be 1:1"

    word_to_wsid_indices = defaultdict(lambda: defaultdict(list))
    word_to_indices = defaultdict(list)
    for i, item in enumerate(train_data):
        word_to_wsid_indices[item["word"]][item["wsid"]].append(i)
        word_to_indices[item["word"]].append(i)

    n = len(train_data)
    n_pos = n_hard_neg = n_easy_neg = skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, item in enumerate(train_data):
            word, wsid = item["word"], item["wsid"]
            query = kb[i]["Question"]

            same_sense = [j for j in word_to_wsid_indices[word][wsid] if j != i]
            if not same_sense:
                skipped += 1
                continue

            pos_j = random.choice(same_sense)
            out.write(json.dumps(
                {"query": query, "candidate": candidate_text(kb[pos_j]), "label": 1},
                ensure_ascii=False) + "\n")
            n_pos += 1

            diff_sense = [j for j in word_to_indices[word] if train_data[j]["wsid"] != wsid]
            if diff_sense:
                neg_j = random.choice(diff_sense)
                out.write(json.dumps(
                    {"query": query, "candidate": candidate_text(kb[neg_j]), "label": 0},
                    ensure_ascii=False) + "\n")
                n_hard_neg += 1

            other_j = random.randrange(n)
            while train_data[other_j]["word"] == word:
                other_j = random.randrange(n)
            out.write(json.dumps(
                {"query": query, "candidate": candidate_text(kb[other_j]), "label": 0},
                ensure_ascii=False) + "\n")
            n_easy_neg += 1

    print(f"Anchors: {n}, skipped (no same-sense alternative): {skipped}")
    print(f"Positive pairs: {n_pos}")
    print(f"Hard-negative pairs (same character, different sense): {n_hard_neg}")
    print(f"Easy-negative pairs (different character): {n_easy_neg}")
    print(f"Total pairs: {n_pos + n_hard_neg + n_easy_neg}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

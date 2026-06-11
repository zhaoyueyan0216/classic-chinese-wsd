"""Convert data/data_result.xlsx into ACP-RAG test_dataset format.

ACP-RAG_Pipeline/01_Context_Selector.py expects two parallel files:
  - a "test_dataset" with one "Question" per sample (same template used for the
    knowledge base, but without revealing the answer)
  - a "Keywords" file with a "、"-separated keyword string per sample, used by
    the pipeline's keyword-matching fallback

We reuse the same 2,000-sample evaluation set (data1/data2, 10 rounds x 100)
already used for the other baselines, keeping gold_wsid/gold_gloss alongside so
predictions can be scored later.
"""

import json

import pandas as pd

INPUT_FILE = "data/data_result.xlsx"
TEST_OUTPUT_FILE = "acprag_baseline/data/test_dataset.json"
KEYWORDS_OUTPUT_FILE = "acprag_baseline/data/test_keywords.json"


def main():
    df = pd.read_excel(INPUT_FILE)
    df = df.drop_duplicates(subset=["dataset", "round", "sample_id"])

    test_dataset = []
    test_keywords = []
    for _, row in df.iterrows():
        word = row["word"]
        context = row["context"]
        test_dataset.append({
            "dataset": row["dataset"],
            "round": int(row["round"]),
            "sample_id": int(row["sample_id"]),
            "word": word,
            "Question": f"古文：「{context}」\n问题：上文中“{word}”字的意思是什么？",
            "gold_wsid": str(row["gold_wsid"]),
            "gold_gloss": row["gold_gloss"],
        })
        test_keywords.append({"Keywords": str(word)})

    with open(TEST_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)
    with open(KEYWORDS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_keywords, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(df)} evaluation rows")
    print(f"Saved {len(test_dataset)} test queries to {TEST_OUTPUT_FILE}")
    print(f"Saved {len(test_keywords)} keyword entries to {KEYWORDS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

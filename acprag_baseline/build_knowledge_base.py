"""Convert data/train_data.json into ACP-QA format for the ACP-RAG baseline.

ACP-RAG (https://github.com/SCUT-DLVCLab/ACP-RAG) expects its knowledge base as a
JSON list of QA records with at least "Id", "Question", "Answer", "Type-A" and
optional "Key*" fields (used by the pipeline's keyword-matching step).

Each (word, context, wsid, gloss) example in train_data.json becomes one QA pair:
the question asks for the meaning of the target character in the given sentence,
and the answer states its gloss. "Type-A" is fixed to "词语解释" (word/character
gloss), one of the task types ACP-RAG's keyword index is built per-task for.
"""

import json

INPUT_FILE = "data/train_data.json"
OUTPUT_FILE = "acprag_baseline/data/acp_qa.json"


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        train_data = json.load(f)

    acp_qa = []
    for i, item in enumerate(train_data):
        word = item["word"]
        context = item["context"]
        gloss = item["gloss"]
        acp_qa.append({
            "Id": i,
            "Question": f"古文：「{context}」\n问题：上文中“{word}”字的意思是什么？",
            "Answer": f"“{word}”在此句中的意思是：{gloss}",
            "Type-A": "词语解释",
            "Key1": word,
            "wsid": item["wsid"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(acp_qa, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(train_data)} train examples")
    print(f"Saved {len(acp_qa)} ACP-QA records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

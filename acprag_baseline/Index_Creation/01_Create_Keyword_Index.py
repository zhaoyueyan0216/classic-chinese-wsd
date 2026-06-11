"""Build Whoosh keyword indexes over the ACP-QA knowledge base.

Adapted from ACP-RAG's Index_Creation/01_Create_Keyword_Index.py:
- reads acprag_baseline/data/acp_qa.json instead of the full ACP-QA corpus
- by default builds an index per distinct "Type-A" value plus an "all" index,
  since ACP-RAG_Pipeline/01_Context_Selector.py looks up index directories by
  both "all" and by the dominant Type-A of the top retrieval results
- "->" in a task name is replaced with "-" so it is a valid directory name
"""

import argparse
import json
import os

from tqdm import tqdm
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in

KB_FILE = "acprag_baseline/data/acp_qa.json"
INDEX_ROOT = "acprag_baseline/index/id_keywords_stash"

schema = Schema(id=ID(stored=True), content=TEXT(stored=True))


def build_index(data, task_name):
    index_dir = os.path.join(INDEX_ROOT, task_name.replace("->", "-"))
    os.makedirs(index_dir, exist_ok=True)

    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for record in tqdm(data, desc=f"Indexing '{task_name}'"):
        writer.add_document(id=str(record["Id"]), content=record["Question"])
    writer.commit()

    print(f"Index for '{task_name}' saved to: {os.path.abspath(index_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=None,
        help='Restrict to a single Type-A value, or "all" for everything. '
        "Default: build one index per distinct Type-A plus an 'all' index.",
    )
    args = parser.parse_args()

    with open(KB_FILE, encoding="utf-8") as f:
        data_all = json.load(f)

    if args.task_name == "all":
        build_index(data_all, "all")
        return

    if args.task_name is not None:
        subset = [r for r in data_all if r["Type-A"] == args.task_name]
        build_index(subset, args.task_name)
        return

    # Default: build "all" plus one index per Type-A
    build_index(data_all, "all")
    task_names = sorted(set(r["Type-A"] for r in data_all))
    for task_name in task_names:
        subset = [r for r in data_all if r["Type-A"] == task_name]
        build_index(subset, task_name)


if __name__ == "__main__":
    main()

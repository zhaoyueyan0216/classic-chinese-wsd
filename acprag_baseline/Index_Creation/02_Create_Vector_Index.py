"""Compute and save FAISS vector index shards for the ACP-QA knowledge base.

Adapted from ACP-RAG's Index_Creation/02_Create_Vector_Index.py:
- embeddings come from this project's GuWenBERT encoder (core/embedding_utils.py,
  768-dim, normalized) instead of ACP-RAG's own 1024-dim embedding model
- texts are encoded in batches via get_embeddings() for GPU throughput
- index shards (IndexFlatL2 over normalized vectors, i.e. cosine similarity) are
  written every --save_interval records, to be merged by 03_Merge_Vector_Index.py
"""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedding_utils import get_embeddings  # noqa: E402

KB_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "acp_qa.json"
OUTPUT_DIR = PROJECT_ROOT / "acprag_baseline" / "index" / "vector_parts"
EMBED_DIM = 768


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_interval", type=int, default=10000,
                         help="Number of records per FAISS shard")
    parser.add_argument("--batch_size", type=int, default=128,
                         help="Encoder batch size")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(KB_FILE, encoding="utf-8") as f:
        data = json.load(f)

    questions = [r["Question"] for r in data]

    part_number = 0
    for start in tqdm(range(0, len(questions), args.save_interval), desc="Encoding shards"):
        batch = questions[start:start + args.save_interval]
        embeddings = get_embeddings(batch, batch_size=args.batch_size)

        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(np.array(embeddings, dtype=np.float32))

        part_number += 1
        out_path = OUTPUT_DIR / f"id_vector_part_{part_number}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(index, f, protocol=4)
        print(f"Saved shard {part_number} ({len(batch)} vectors) to {out_path}")

    print(f"Done. {part_number} shard(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

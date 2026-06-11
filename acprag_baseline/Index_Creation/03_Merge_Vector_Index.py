"""Merge all FAISS vector index shards into one index.

Adapted from ACP-RAG's Index_Creation/03_Merge_Vector_Index.py: instead of a
hardcoded part count, this globs all id_vector_part_*.pkl shards produced by
02_Create_Vector_Index.py.
"""

import pickle
from pathlib import Path

import faiss
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = PROJECT_ROOT / "acprag_baseline" / "index" / "vector_parts"
OUTPUT_PATH = PROJECT_ROOT / "acprag_baseline" / "index" / "acp_qa_vector.index"


def main():
    shard_paths = sorted(
        SHARD_DIR.glob("id_vector_part_*.pkl"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {SHARD_DIR}")

    indexes = []
    for path in tqdm(shard_paths, desc="Loading shards"):
        with open(path, "rb") as f:
            indexes.append(pickle.load(f))

    index_dim = indexes[0].d
    merged_index = faiss.IndexFlatL2(index_dim)
    for idx in tqdm(indexes, desc="Merging"):
        vectors = idx.reconstruct_n(0, idx.ntotal)
        merged_index.add(vectors)

    faiss.write_index(merged_index, str(OUTPUT_PATH))
    print(f"Merged {merged_index.ntotal} vectors (dim={index_dim}) into {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

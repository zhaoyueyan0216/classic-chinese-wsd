"""Select candidate knowledge-base contexts for each test query.

Adapted from ACP-RAG_Pipeline/01_Context_Selector.py. The original combines a
FAISS vector search with a cross-encoder rank model and a multi-stage,
Type-A-aware Whoosh fallback. Our knowledge base has a single Type-A
("词语解释"), so the Whoosh fallback collapses to a simple keyword bonus, and
the rank model is the GuWenBERT cross-encoder fine-tuned by
Model_Training/train_rank_model.py (see build_rank_pairs.py for how its
training pairs are constructed):

1. FAISS top-N search with the GuWenBERT query embedding (cosine similarity,
   recovered from IndexFlatL2 distances over normalized vectors) gives a
   candidate pool.
2. The fine-tuned cross-encoder reranker scores each (query, candidate) pair;
   if it hasn't been trained yet, falls back to the FAISS cosine similarity.
3. A keyword bonus: knowledge-base records whose target character ("Key1")
   matches the query's target character (from test_keywords.json) get a score
   boost, mirroring ACP-RAG's keyword-match fallback.
4. Top-5 by combined score, written to "检索结果".
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedding_utils import get_embedding  # noqa: E402

KB_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "acp_qa.json"
TEST_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "test_dataset.json"
KEYWORDS_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "test_keywords.json"
VECTOR_INDEX_FILE = PROJECT_ROOT / "acprag_baseline" / "index" / "acp_qa_vector.index"
KEYWORD_INDEX_DIR = PROJECT_ROOT / "acprag_baseline" / "index" / "id_keywords_stash" / "all"
RANK_MODEL_DIR = PROJECT_ROOT / "acprag_baseline" / "Model_Training" / "rank_model"
OUTPUT_FILE = PROJECT_ROOT / "acprag_baseline" / "results" / "01_context_selected.json"

VECTOR_TOP_N = 30
FINAL_TOP_K = 5
KEYWORD_BONUS = 0.5
RERANK_MAX_LENGTH = 256


def cosine_from_l2(distances):
    """IndexFlatL2 over unit-normalized vectors: ||a-b||^2 = 2 - 2*cos(a,b)."""
    return 1.0 - np.asarray(distances) / 2.0


def candidate_text(record):
    return f"问题：{record['Question']}答案：{record['Answer']}"


def keyword_search(ix, word):
    parser = QueryParser("content", ix.schema)
    query = parser.parse(word)
    matched_ids = set()
    with ix.searcher() as searcher:
        for hit in searcher.search(query, limit=None):
            matched_ids.add(int(hit["id"]))
    return matched_ids


def load_rank_model():
    if not (RANK_MODEL_DIR / "config.json").exists():
        print(f"No fine-tuned rank model found at {RANK_MODEL_DIR}; "
              f"falling back to FAISS cosine similarity for ranking.")
        return None, None, None

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(RANK_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(RANK_MODEL_DIR))
    model.to(device)
    model.eval()
    print(f"Loaded rank model from {RANK_MODEL_DIR} ({device})")
    return tokenizer, model, device


@torch.no_grad()
def rerank_scores(query, candidates, tokenizer, model, device):
    """Return P(relevant) for each (query, candidate) pair."""
    inputs = tokenizer(
        [query] * len(candidates), candidates,
        padding=True, truncation=True, max_length=RERANK_MAX_LENGTH, return_tensors="pt",
    ).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return probs.cpu().tolist()


def main():
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    id_to_record = {r["Id"]: r for r in kb}

    with open(TEST_FILE, encoding="utf-8") as f:
        test_data = json.load(f)
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        test_keywords = json.load(f)

    faiss_index = faiss.read_index(str(VECTOR_INDEX_FILE))
    keyword_ix = open_dir(str(KEYWORD_INDEX_DIR))
    rank_tokenizer, rank_model, rank_device = load_rank_model()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for sample, kw in tqdm(zip(test_data, test_keywords), total=len(test_data), desc="Retrieving"):
        query = sample["Question"]
        target_word = kw["Keywords"]

        embedding = get_embedding(query).astype(np.float32).reshape(1, -1)
        distances, indices = faiss_index.search(embedding, VECTOR_TOP_N)
        indices = [int(idx) for idx in indices[0]]

        if rank_model is not None:
            candidates = [candidate_text(id_to_record[idx]) for idx in indices]
            base_scores = rerank_scores(query, candidates, rank_tokenizer, rank_model, rank_device)
        else:
            base_scores = cosine_from_l2(distances[0]).tolist()

        matched_ids = keyword_search(keyword_ix, target_word)

        scored = []
        for idx, base_score in zip(indices, base_scores):
            score = float(base_score)
            if idx in matched_ids or id_to_record.get(idx, {}).get("Key1") == target_word:
                score += KEYWORD_BONUS
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:FINAL_TOP_K]

        sample_out = dict(sample)
        sample_out["检索结果"] = [[idx, round(score, 4)] for idx, score in top_k]
        results.append(sample_out)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

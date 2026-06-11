# ACP-RAG baseline

Adaptation of [SCUT-DLVCLab/ACP-RAG](https://github.com/SCUT-DLVCLab/ACP-RAG) for the
single-character WSD task, using this project's data and LLM/embedding setup
instead of ACP-RAG's own corpus and trained models.

Differences from upstream ACP-RAG:
- Knowledge base = `data/train_data.json` (90,131 word/context/gloss examples,
  with the 2,000 evaluation contexts already excluded), reformatted as ACP-QA
  records with `Type-A = "词语解释"`.
- Embeddings: GuWenBERT (`core/embedding_utils.py`, 768-dim) instead of
  ACP-RAG's 1024-dim embedding model.
- Context filter / response generation: the project's DashScope chat LLM
  (`acprag_baseline/common.py`, `core/llm.py`) instead of locally-hosted
  Qwen checkpoints.
- Response generation outputs a `wsid` selection (not free text): the LLM is
  shown the target character's full candidate-sense inventory
  (`data/character_senses.json`, `wsid=X: gloss`) plus the filtered in-context
  QA examples, and must return strict JSON `{"prediction": "wsid", "reason":
  "..."}` -- the same pattern as `experiments/main_experiment.py`, so accuracy
  is directly comparable to the other methods in `data/data_result.xlsx`.
- Cross-encoder rank model: GuWenBERT fine-tuned on (query, candidate) pairs
  built from `data/train_data.json` (see `Model_Training/`), instead of
  ACP-RAG's own BERT_guwen rank model (not released). Candidate ranking
  combines this reranker's score with a keyword-match bonus; if the model
  hasn't been trained yet, falls back to FAISS cosine similarity.

Requires `faiss-cpu` and `Whoosh` in addition to this project's existing deps.

## Pipeline

Run from the project root.

```bash
# 0. Build the QA knowledge base and evaluation queries
python acprag_baseline/build_knowledge_base.py
python acprag_baseline/build_test_set.py

# 1. Build Whoosh keyword index
python "acprag_baseline/Index_Creation/01_Create_Keyword_Index.py"

# 2. Compute FAISS vector index shards (GuWenBERT embeddings)
python "acprag_baseline/Index_Creation/02_Create_Vector_Index.py"

# 3. Merge shards into one FAISS index
python "acprag_baseline/Index_Creation/03_Merge_Vector_Index.py"

# 3.5 (optional but recommended) Train the cross-encoder reranker
python "acprag_baseline/Model_Training/build_rank_pairs.py"
python "acprag_baseline/Model_Training/train_rank_model.py"

# 4. Retrieve candidate contexts (vector search + reranker + keyword bonus)
python "acprag_baseline/ACP-RAG_Pipeline/01_Context_Selector.py"

# 5. LLM-based relevance filtering
python "acprag_baseline/ACP-RAG_Pipeline/02_Context_Filter.py"

# 6. Generate final WSD responses
python "acprag_baseline/ACP-RAG_Pipeline/03_Response_Generation.py"
```

Outputs (gitignored, generated locally):
- `acprag_baseline/data/acp_qa.json` — knowledge base
- `acprag_baseline/data/test_dataset.json`, `test_keywords.json` — evaluation queries
- `acprag_baseline/data/rank_pairs.jsonl` — cross-encoder training pairs
- `acprag_baseline/Model_Training/rank_model/` — fine-tuned reranker checkpoint
- `acprag_baseline/index/` — Whoosh + FAISS indexes
- `acprag_baseline/results/` — per-step pipeline outputs, ending in
  `03_response_generated.json` (one `predicted_wsid`/`predicted_gloss`/
  `reason`/`correct` per evaluation sample, alongside the retained
  `gold_wsid`/`gold_gloss`, so accuracy can be computed directly)

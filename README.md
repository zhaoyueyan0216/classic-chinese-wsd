# Ancient Chinese Word Sense Disambiguation via Multi-Agent RAG

Code for the paper: *[your paper title]*.

The pipeline disambiguates Classical Chinese word senses by retrieving historical usage evidence from a large corpus and coordinating multiple LLM agents to reason over that evidence.

## How it works

Given a target word in context, the pipeline runs in five stages:

1. **Top-K filtering** — the LLM votes three times independently; we take the union of selected senses as candidates
2. **Pseudo query generation** — for each candidate sense, generate pseudo Classical Chinese sentences to use as BM25 queries
3. **BM25 retrieval + reranking** — retrieve from the corpus, rerank by embedding similarity to the sense gloss
4. **Sense agent verification** — a per-sense LLM agent filters noisy sentences and assesses contextual fit
5. **Aggregator decision** — a final LLM weighs all agent outputs under an explicit evidence-bias correction protocol

## Requirements

- Python 3.9+
- Java 11+ (required by Pyserini)
- GPU recommended for embedding (CPU works but is slow)

```bash
conda create -n wsd python=3.9
conda activate wsd
pip install -r requirements.txt
pip install pyserini
```

Verify Java: `java -version`

## Setup

### 1. Config

```bash
cp config.yaml.example config.yaml
```

Fill in your API key in `config.yaml`. The default setup uses DeepSeek via DashScope:

```yaml
api:
  api_key: "YOUR_API_KEY_HERE"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
models:
  llm_model: "deepseek-v3.2-exp"
```

Any OpenAI-compatible endpoint works — change `base_url` and `llm_model` accordingly. `config.yaml` is gitignored.

### 2. GuWenBERT

The embedding module uses [GuWenBERT](https://huggingface.co/ethanyt/guwenbert-base). It downloads automatically on first run. On an HPC cluster without internet, download it first then point to the local path:

```bash
export GUWENBERT_MODEL_PATH=/path/to/guwenbert-base
```

### 3. Build the BM25 index

The index is built from `retrieval_corpus/` and is not in the repository (14 GB). Build it once:

```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input retrieval_corpus/ \
  --index bm25_index_zh/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --language zh
```

### 4. Smoke test

```bash
python experiments/main_experiment.py --rounds 1 --samples 2
```

This runs 2 samples through the full pipeline. If it completes without error, everything is wired up correctly.

## Running experiments

All commands are run from the project root.

**Full pipeline:**
```bash
python experiments/main_experiment.py --rounds 20 --seed 42
```

**Ablation studies** (each reads `dataprocess/data/data1.xlsx` and `data2.xlsx`):
```bash
python experiments/ablation_a_experiment.py --target both        # remove LLM verification
python experiments/ablation_b_experiment.py --target both        # remove aggregator
python experiments/ablation_nohistory_experiment.py --target both # remove retrieval entirely
python experiments/ablation_nopseudo_experiment.py --rounds 20   # remove pseudo query generation
python experiments/naive_rag_experiment.py --target both         # naive RAG baseline
```

Results are written to timestamped `.xlsx` files in the project root.

## Caching

LLM calls (Top-K and sense agent) are cached to `llm_cache/`. On repeated runs, cached calls are skipped. The aggregator is intentionally not cached — its non-determinism is the source of variance across the 20 rounds used for significance testing.

## Repository structure

```
core/                   LLM wrapper, embedding encoder, BM25 retrieval
experiments/            All experiment and ablation scripts
analysis/               Plotting and result analysis scripts
data/                   Sense definitions and annotated contexts
dataprocess/data/       Evaluation test sets (data1.xlsx, data2.xlsx)
retrieval_corpus/       Classical Chinese corpus (input for BM25 indexing)
bm25_index_zh/          BM25 index — built locally, not in git
```

## Troubleshooting

`RuntimeError: LLM调用彻底失败: 401` — API key is wrong or expired.

`FileNotFoundError: bm25_index_zh` — index not built yet, see step 3.

`OSError: Can't load tokenizer for 'ethanyt/guwenbert-base'` — set `GUWENBERT_MODEL_PATH` to a local snapshot.

`ModuleNotFoundError: pyserini` — run `pip install pyserini` and check that Java 11+ is on your PATH.

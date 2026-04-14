# Ancient Chinese Word Sense Disambiguation System

## Project Overview

This project implements a complete pipeline for ancient Chinese word sense disambiguation (WSD) using pseudo-ancient Chinese generation, Pyserini BM25 retrieval, and LLM-based decision making. The system is designed to accurately identify the correct sense of ancient Chinese words in context.

## System Architecture

The system consists of the following components:

1. **Pseudo-Ancient Chinese Generation** - Generates pseudo-ancient Chinese examples for better retrieval
2. **Pyserini BM25 Retrieval** - Retrieves relevant supporting sentences from a corpus
3. **Embedding Reranking** - Reranks retrieved sentences based on semantic similarity
4. **LLM-based Top-k Filtering** - Uses LLM to filter top-k candidate senses
5. **Historical Evidence Construction** - Builds evidence from retrieved sentences
6. **Final LLM Decision** - Makes the final sense disambiguation decision

## Project Structure

```
testbm25pipeline/
├── main.py                # Main test script
├── pseudo_retrieval.py     # Pseudo-ancient Chinese generation and retrieval
├── embedding_utils.py      # Embedding utilities
├── llm.py                 # LLM interaction utilities
├── prototype.py           # Prototype vector construction
├── schema.py              # Usage schema construction
├── decision.py            # Decision utilities
├── requirements.txt       # Project dependencies
├── data/                  # Test data
│   ├── character/         # Character-level data
│   └── compound/          # Compound word data
├── guwenbert_output/      # GuwenBERT model output
└── bm25_index_zh/         # BM25 index directory
```

## Prerequisites

- Python 3.8+
- PIP package manager
- Access to LLM API (e.g., OpenAI, DeepSeek)
- Pyserini for BM25 retrieval
- Pre-built BM25 index for ancient Chinese corpus


3. Set up environment variables for LLM API access :

```bash
# For example, for OpenAI
export OPENAI_API_KEY=your-api-key

# For DeepSeek
export DEEPSEEK_API_KEY=your-api-key
```

4. Ensure the BM25 index is available at the specified path in `main.py`:

```python
index_dir = r"/mimer/NOBACKUP/groups/cik_data/yueyan/testfrozen_pipeline/bm25_index_zh"
```

## Usage

### Running the Full System

To run the full system with all components enabled:

```bash
python main.py
```

### Running Ablation Experiments

The system supports several ablation experiments to evaluate the contribution of different components:

```bash
# No Top-2 filtering
python main.py ablation_no_top2

# Only Top-2 filtering (no other components)
python main.py ablation_top2_only

# No historical evidence
python main.py ablation_no_history

# No pseudo-ancient Chinese generation
python main.py ablation_no_pseudo

# No usage schema
python main.py ablation_no_schema

# No similarity calculation
python main.py ablation_no_similarity
```

## Configuration Options

The system is configured through the `BASE_CONFIG` dictionary in `main.py`:

| Configuration Option | Description | Default Value |
|---------------------|-------------|---------------|
| `use_top2_filter` | Use LLM to filter top-2 candidate senses | True |
| `use_pseudo_query` | Use pseudo-ancient Chinese generation | True |
| `use_bm25` | Use Pyserini BM25 retrieval | True |
| `use_rerank` | Use embedding-based reranking | True |
| `use_schema` | Use usage schema construction | True |
| `use_similarity` | Use similarity calculation | True |
| `use_supporting_sentences` | Use supporting sentences | True |
| `final_decision_with_llm` | Use LLM for final decision | True |
| `use_final_cache` | Use cache for final LLM decisions | False |

## Output

The system generates detailed test results in Excel or CSV format, including:

- Target word
- Context sentence
- Gold sense ID and definition
- Predicted sense ID
- Prediction result (Correct/Incorrect)
- Excluded senses
- Decision reasoning
- Number of candidate senses
- LLM-generated Top-2 senses

## Evaluation Metrics

The system evaluates performance using accuracy, calculated as:

```
Accuracy = Number of correct predictions / Total number of test cases
```

## Sample Output

```
Ancient Chinese Word Sense Disambiguation System Test - Using Pseudo-Ancient Chinese Generation
Configuration: full
==========================================================================================
Configuration details:
  use_top2_filter: True
  use_pseudo_query: True
  use_bm25: True
  use_rerank: True
  use_schema: True
  use_similarity: True
  use_supporting_sentences: True
  final_decision_with_llm: True
  use_final_cache: False
==========================================================================================
Loaded 1000 test contexts
Using 999 senses for testing
Actual number of target words to test: 99 (sorted)
Sorted test words: ['严', '丹', '体', '余', '倕', '偶', '入', '冲', '凉', '劳', ...]

>>> Starting batch processing, total 10 batches, 100 test cases per batch

=== Batch 1 completed, cumulative result: 85/100 = 0.850 ===
=== Batch 2 completed, cumulative result: 172/200 = 0.860 ===
...

>>> Exporting test results to Excel
✅ Successfully exported test results to pseudo_test_results_full_20260414_123456.xlsx
Exported 999 test records
Added statistics row: Number of test cases=999, Number of correct predictions=867, Accuracy=0.868

==========================================================================================
Test completed: 867/999 = 0.868
==========================================================================================
```

#

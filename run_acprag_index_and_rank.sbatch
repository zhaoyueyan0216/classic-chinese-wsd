#!/bin/bash
#SBATCH -J acprag_index_rank
#SBATCH -A naiss2026-3-306
#SBATCH -p alvis
#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:T4:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

echo "===== JOB START ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

cd /mimer/NOBACKUP/groups/cik_data/yueyan/testfrozen_pipeline
mkdir -p logs

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load Java/21.0.8

source /mimer/NOBACKUP/groups/cik_data/yueyan/envs/pipeline_env/bin/activate

echo "Python path:"
which python
echo "Python version:"
python --version

export HF_HOME=/mimer/NOBACKUP/groups/cik_data/yueyan/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export GUWENBERT_MODEL_PATH=/mimer/NOBACKUP/groups/cik_data/yueyan/testfrozen_pipeline/llm_cache/models--ethanyt--guwenbert-base/snapshots/eff0d4a5196d7bf7b8be746c5c6437e89d8b9061

mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

echo "HF_HOME: $HF_HOME"
echo "GUWENBERT_MODEL_PATH: $GUWENBERT_MODEL_PATH"
echo "Current dir: $(pwd)"

echo "===== GPU INFO ====="
nvidia-smi || true
python -c "import torch; print('cuda available =', torch.cuda.is_available())"

echo "===== Installing extra deps (faiss-cpu, Whoosh) ====="
pip install -q faiss-cpu Whoosh

echo "===== STEP 1: Build Whoosh keyword index ====="
python "acprag_baseline/Index_Creation/01_Create_Keyword_Index.py"

echo "===== STEP 2: Build FAISS vector index shards (GuWenBERT) ====="
python "acprag_baseline/Index_Creation/02_Create_Vector_Index.py"

echo "===== STEP 3: Merge FAISS vector index shards ====="
python "acprag_baseline/Index_Creation/03_Merge_Vector_Index.py"

echo "===== STEP 4: Train cross-encoder rank model ====="
python "acprag_baseline/Model_Training/train_rank_model.py"

echo "===== JOB FINISHED ====="
echo "End time: $(date)"

"""Fine-tune GuWenBERT as a 2-way cross-encoder reranker.

Adapted from ACP-RAG's Model_Training/Rank_Model_Training.py: trains
AutoModelForSequenceClassification (relevant / not relevant) on
(query, candidate) pairs from rank_pairs.jsonl (see build_rank_pairs.py),
producing the reranker used by ACP-RAG_Pipeline/01_Context_Selector.py.

Differences from the original script: single-GPU, full fine-tuning (no frozen
layers), standard train/val split with best-checkpoint saving, and the base
model is GuWenBERT (same encoder already used for retrieval embeddings) rather
than ACP-RAG's own BERT_guwen checkpoint.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIRS_FILE = PROJECT_ROOT / "acprag_baseline" / "data" / "rank_pairs.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "acprag_baseline" / "Model_Training" / "rank_model"

DEFAULT_MODEL_NAME = os.environ.get("GUWENBERT_MODEL_PATH", "ethanyt/guwenbert-base")


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_collate_fn(tokenizer, max_length):
    def collate(batch):
        queries = [b["query"] for b in batch]
        candidates = [b["candidate"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = tokenizer(queries, candidates, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt")
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item() * batch["labels"].size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(PAIRS_FILE, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    random.shuffle(pairs)

    n_val = int(len(pairs) * args.val_ratio)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME, num_labels=2, local_files_only=True, use_safetensors=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(PairDataset(train_pairs), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running_loss / len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Saved best model (val_loss={val_loss:.4f}) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

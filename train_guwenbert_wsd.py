import os
import json
import math
import random
import argparse
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

DEFAULT_MODEL_NAME = "/mimer/NOBACKUP/groups/cik_data/yueyan/testfrozen_pipeline/llm_cache/models--ethanyt--guwenbert-base/snapshots/eff0d4a5196d7bf7b8be746c5c6437e89d8b9061"


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_maybe_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "RECORDS" in data:
        data = data["RECORDS"]
    return data


def normalize_wsid(x):
    return str(x).strip() if x is not None else None


def mark_target_word(text: str, word: str) -> str:
    """
    用普通符号标目标词，避免引入特殊token。
    只标第一次出现，先跑通实验。
    """
    if not text or not word:
        return text
    if word in text:
        return text.replace(word, f"【{word}】", 1)
    return text


def build_sense_mappings(senses):
    wsid_to_sense = {}
    word_to_senses = defaultdict(list)

    for entry in senses:
        if not isinstance(entry, dict):
            continue
        wsid = normalize_wsid(entry.get("wsid"))
        word = entry.get("word")
        gloss = entry.get("newgloss")
        if not wsid or not word or not gloss:
            continue

        item = {
            "wsid": wsid,
            "word": word,
            "newgloss": gloss
        }
        wsid_to_sense[wsid] = item
        word_to_senses[word].append(item)

    return wsid_to_sense, dict(word_to_senses)


def group_pair_data(pair_data):
    """
    按原始实例分组，避免同一条句子的不同候选pair被拆到train/dev两边。
    key = (text, word, gold_wsid)
    """
    groups = defaultdict(list)
    for item in pair_data:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        word = item.get("word")
        gold_wsid = normalize_wsid(item.get("gold_wsid"))
        if not text or not word or not gold_wsid:
            continue
        key = (text, word, gold_wsid)
        groups[key].append(item)
    return groups


def split_grouped_pair_data(pair_data, dev_ratio=0.1, seed=42):
    groups = group_pair_data(pair_data)
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    dev_size = max(1, int(len(keys) * dev_ratio))
    dev_keys = set(keys[:dev_size])

    train_data = []
    dev_data = []

    for k, items in groups.items():
        if k in dev_keys:
            dev_data.extend(items)
        else:
            train_data.extend(items)

    return train_data, dev_data


class PairDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]
        word = item["word"]
        gloss = item["gloss"]
        label = int(item["label"])

        marked_text = mark_target_word(text, word)

        enc = self.tokenizer(
            marked_text,
            gloss,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_binary_metrics(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_model(
    model,
    tokenizer,
    train_data,
    dev_data,
    output_dir,
    batch_size=16,
    max_length=256,
    lr=2e-5,
    epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    device="cpu",
    checkpoint_interval=20000,  # 每500步保存一次checkpoint
):
    train_dataset = PairDataset(train_data, tokenizer, max_length=max_length)
    dev_dataset = PairDataset(dev_data, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_dev_acc = -1.0
    best_model_path = os.path.join(output_dir, "best_model")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 检查是否有checkpoint
    latest_checkpoint = None
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_"):
            latest_checkpoint = os.path.join(checkpoint_dir, file)
            break
    
    start_epoch = 0
    if latest_checkpoint:
        print(f"Loading checkpoint from: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dev_acc = checkpoint.get("best_dev_acc", best_dev_acc)
        print(f"Resuming training from epoch {start_epoch}")

    model.to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

            if step % 200 == 0:
                print(
                    f"Epoch {epoch+1} | step {step}/{len(train_loader)} "
                    f"| train_loss={running_loss / max(seen,1):.4f}"
                )

            # 保存checkpoint
            if step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}_step{step}.pt")
                checkpoint = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dev_acc": best_dev_acc,
                    "loss": running_loss / max(seen, 1)
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")

        train_loss = running_loss / max(seen, 1)
        dev_loss, dev_acc = compute_binary_metrics(model, dev_loader, device)

        print(
            f"\n[Epoch {epoch+1}] "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_acc:.4f}"
        )

        # 保存epoch checkpoint
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dev_acc": best_dev_acc,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc
        }
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Epoch checkpoint saved to: {epoch_checkpoint_path}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print(f"New best model: dev_acc={dev_acc:.4f}, saving to {best_model_path}")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

    print(f"\nBest dev acc = {best_dev_acc:.4f}")
    return best_model_path


def score_candidate_batch(model, tokenizer, text, word, candidate_senses, device, max_length=256):
    """
    给一个测试实例的所有候选义项打分。
    返回按分数从高到低排序后的列表。
    """
    model.eval()

    marked_text = mark_target_word(text, word)
    texts_a = [marked_text] * len(candidate_senses)
    texts_b = [sense["newgloss"] for sense in candidate_senses]

    enc = tokenizer(
        texts_a,
        texts_b,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].tolist()

    scored = []
    for sense, score in zip(candidate_senses, probs):
        scored.append({
            "wsid": normalize_wsid(sense["wsid"]),
            "gloss": sense["newgloss"],
            "score": float(score)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def evaluate_wsd(
    model,
    tokenizer,
    test_contexts,
    wsid_to_sense,
    word_to_senses,
    device,
    max_length=256,
    topk_list=(1, 3, 5),
):
    total = 0
    correct = 0
    reciprocal_rank_sum = 0.0
    recall_hits = {k: 0 for k in topk_list}
    results = []

    for idx, ctx in enumerate(test_contexts, start=1):
        gold_wsid = normalize_wsid(ctx.get("wsid"))
        text = ctx.get("txt")

        if not gold_wsid or not text:
            continue
        if gold_wsid not in wsid_to_sense:
            continue

        word = wsid_to_sense[gold_wsid]["word"]
        candidate_senses = word_to_senses.get(word, [])
        if len(candidate_senses) < 2:
            continue

        scored = score_candidate_batch(
            model=model,
            tokenizer=tokenizer,
            text=text,
            word=word,
            candidate_senses=candidate_senses,
            device=device,
            max_length=max_length,
        )

        total += 1
        pred_wsid = scored[0]["wsid"]
        is_correct = pred_wsid == gold_wsid
        if is_correct:
            correct += 1

        rank = None
        for i, item in enumerate(scored, start=1):
            if item["wsid"] == gold_wsid:
                rank = i
                reciprocal_rank_sum += 1.0 / i
                break

        for k in topk_list:
            topk_wsids = [x["wsid"] for x in scored[:k]]
            if gold_wsid in topk_wsids:
                recall_hits[k] += 1

        results.append({
            "idx": idx,
            "word": word,
            "text": text,
            "gold_wsid": gold_wsid,
            "gold_gloss": wsid_to_sense[gold_wsid]["newgloss"],
            "pred_wsid": pred_wsid,
            "pred_gloss": scored[0]["gloss"],
            "correct": is_correct,
            "rank": rank,
            "top_candidates": scored[:5],
        })

        if idx % 100 == 0:
            print(f"Evaluated {idx}/{len(test_contexts)} test contexts...")

    metrics = {
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "mrr": reciprocal_rank_sum / total if total > 0 else 0.0,
    }
    for k in topk_list:
        metrics[f"recall@{k}"] = recall_hits[k] / total if total > 0 else 0.0

    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_train_json", type=str, required=True, help="training_data_pair.json")
    parser.add_argument("--test_context_json", type=str, required=True, help="1000条测试集context json")
    parser.add_argument("--senses_json", type=str, required=True, help="senses.json")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="guwenbert_wsd_output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n=== Starting training script ===")
    print(f"Command line arguments: {args}")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")

    print("\n1. Loading senses...")
    print(f"Senses file: {args.senses_json}")
    senses = load_json_maybe_records(args.senses_json)
    print(f"Loaded raw senses data: {len(senses)} items")
    wsid_to_sense, word_to_senses = build_sense_mappings(senses)
    print(f"Built sense mappings: {len(wsid_to_sense)} senses, {len(word_to_senses)} words")

    print("\n2. Loading pair training data...")
    print(f"Training data file: {args.pair_train_json}")
    pair_data = load_json_maybe_records(args.pair_train_json)
    print(f"Loaded pair training data: {len(pair_data)} items (using first 500 items for testing)")

    print("\n3. Splitting train/dev by grouped instances...")
    train_data, dev_data = split_grouped_pair_data(
        pair_data,
        dev_ratio=args.dev_ratio,
        seed=args.seed
    )
    print(f"Split completed: {len(train_data)} train pairs, {len(dev_data)} dev pairs")

    print("\n4. Loading tokenizer/model...")
    print(f"Model name: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Loaded tokenizer: {type(tokenizer).__name__}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    print(f"Loaded model: {type(model).__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n5. Starting model training...")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, Max length: {args.max_length}")
    best_model_path = train_model(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        dev_data=dev_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
    )
    print(f"Training completed. Best model saved to: {best_model_path}")

    print("\n6. Reloading best model for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)
    print("Model reloaded successfully")

    print("\n7. Loading test contexts...")
    print(f"Test data file: {args.test_context_json}")
    test_contexts = load_json_maybe_records(args.test_context_json)
    print(f"Loaded test contexts: {len(test_contexts)} items")

    print("\n8. Running WSD evaluation on test set...")
    metrics, results = evaluate_wsd(
        model=model,
        tokenizer=tokenizer,
        test_contexts=test_contexts,
        wsid_to_sense=wsid_to_sense,
        word_to_senses=word_to_senses,
        device=device,
        max_length=args.max_length,
        topk_list=(1, 3, 5),
    )

    print("\n=== Final WSD Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    results_path = os.path.join(args.output_dir, "test_predictions.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {results_path}")
    print("\n=== Training script completed ===")


if __name__ == "__main__":
    main()
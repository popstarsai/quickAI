"""Train two independent nano language models within a shared time budget."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .model import CharTokenizer, ModelConfig, TinyTransformer


ROOT = Path(__file__).resolve().parent.parent
CORPORA = {"fields": ROOT / "data" / "fields.txt", "stories": ROOT / "data" / "stories.txt"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_corpus(text: str, tokenizer: CharTokenizer, block_size: int):
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    boundary = int(len(tokens) * 0.9)
    train, val = tokens[:boundary], tokens[boundary:]
    if min(len(train), len(val)) <= block_size:
        raise ValueError(
            f"Corpus too short for block_size={block_size}; need >{block_size} characters "
            "in both 90/10 partitions. Lower --block-size or provide more data."
        )
    return train, val


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    offsets = torch.randint(0, len(data) - block_size, (batch_size,))
    xs = torch.stack([data[i:i + block_size] for i in offsets])
    ys = torch.stack([data[i + 1:i + block_size + 1] for i in offsets])
    return xs.to(device), ys.to(device)


@torch.no_grad()
def evaluate(model, train_data, val_data, batch_size, block_size, device, batches):
    model.eval()
    scores = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        scores[split] = sum(losses) / len(losses)
    model.train()
    return scores


def train_one(name: str, corpus_path: Path, args, device: str, deadline: float):
    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty corpus: {corpus_path}")
    tokenizer = CharTokenizer.from_text(text)
    train_data, val_data = split_corpus(text, tokenizer, args.block_size)
    config = ModelConfig(
        vocab_size=len(tokenizer.chars), block_size=args.block_size,
        n_embd=args.embed_dim, n_head=args.heads, n_layer=args.layers,
        dropout=args.dropout,
    )
    torch.manual_seed(args.seed + list(CORPORA).index(name))
    model = TinyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    params = sum(p.numel() for p in model.parameters())
    initial = evaluate(model, train_data, val_data, args.batch_size,
                       args.block_size, device, args.eval_batches)
    start = time.monotonic()
    history = []
    steps_done = 0
    for step in range(args.steps):
        if time.monotonic() >= deadline:
            break
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps_done += 1
        if step == 0 or steps_done % max(1, args.steps // 5) == 0:
            history.append({"step": steps_done, "batch_train_loss": round(loss.item(), 6)})

    final = evaluate(model, train_data, val_data, args.batch_size,
                     args.block_size, device, args.eval_batches)
    model.eval()
    initial_token = tokenizer.encode(text[0])[0]
    sample = tokenizer.decode(model.generate(
        torch.tensor([[initial_token]], dtype=torch.long, device=device),
        args.sample_tokens,
    )[0].tolist())
    folder = args.out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "config": asdict(config),
        "chars": tokenizer.chars, "start_char": text[0],
    }, folder / "checkpoint.pt")
    (folder / "sample.txt").write_text(sample, encoding="utf-8")
    metrics = {
        "model": name,
        "dataset": str(corpus_path),
        "dataset_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text), "vocab_size": config.vocab_size,
        "parameters": params, "device": device, "seed": args.seed + list(CORPORA).index(name),
        "steps_completed": steps_done, "steps_requested": args.steps,
        "training_seconds": round(time.monotonic() - start, 3),
        "initial_loss": initial, "final_loss": final, "loss_history": history,
        "config": asdict(config),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {steps_done}/{args.steps} steps | {params:,} params | "
          f"validation loss {initial['val']:.3f} -> {final['val']:.3f} | "
          f"{metrics['training_seconds']}s", flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=20.0, help="Total wall-clock budget shared by both models")
    parser.add_argument("--steps", type=int, default=300, help="Maximum optimizer steps per model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--fields-data", type=Path, default=CORPORA["fields"])
    parser.add_argument("--stories-data", type=Path, default=CORPORA["stories"])
    args = parser.parse_args(argv)
    if args.minutes <= 0 or args.steps <= 0 or args.batch_size <= 0:
        parser.error("minutes, steps and batch-size must be positive")
    if args.eval_batches <= 0 or args.sample_tokens < 0 or args.learning_rate <= 0:
        parser.error("eval-batches, learning-rate must be positive; sample-tokens >= 0")

    device = choose_device(args.device)
    if device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"QuickAI | device={device} | total budget={args.minutes:g} minutes", flush=True)
    start = time.monotonic()
    deadline = start + args.minutes * 60
    inputs = (("fields", args.fields_data), ("stories", args.stories_data))
    results = []
    for index, (name, path) in enumerate(inputs):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise RuntimeError(f"Budget exhausted before {name}; earlier results are saved")
        slice_deadline = time.monotonic() + remaining_seconds / (len(inputs) - index)
        print(f"Training {name} from {path}...", flush=True)
        results.append(train_one(name, path, args, device, slice_deadline))
    summary = {
        "total_seconds": round(time.monotonic() - start, 3),
        "budget_minutes": args.minutes,
        "models": [{"name": r["model"], "steps": r["steps_completed"],
                    "parameters": r["parameters"], "val_loss": r["final_loss"]["val"]}
                   for r in results],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

"""Generate text from a QuickAI checkpoint (never executes an untrusted pickle)."""

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from .model import CharTokenizer, ModelConfig, TinyTransformer


def load_model(path: Path, device: str = "cpu"):
    bundle = torch.load(path, map_location=device, weights_only=True)
    config_fields = {field.name for field in fields(ModelConfig)}
    config = ModelConfig(**{key: value for key, value in bundle["config"].items()
                            if key in config_fields})
    tokenizer = CharTokenizer(bundle["chars"])
    model = TinyTransformer(config).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, tokenizer, bundle["start_char"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args(argv)
    model, tokenizer, start_char = load_model(args.checkpoint, args.device)
    prefix = start_char if args.prompt is None else args.prompt
    if not prefix:
        parser.error("Prompt cannot be empty")
    tokens = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=args.device)
    result = model.generate(tokens, args.tokens, args.temperature)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()

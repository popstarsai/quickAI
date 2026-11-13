"""Fast correctness and CLI tests; no network or GPU required."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from quickai.generate import load_model
from quickai.model import CharTokenizer, ModelConfig, TinyTransformer
from quickai.train import main, split_corpus


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.config = ModelConfig(vocab_size=10, block_size=8, n_embd=16,
                                  n_head=2, n_layer=1, dropout=0.0)
        self.model = TinyTransformer(self.config).eval()

    def test_tokenizer_roundtrip_and_unknown(self):
        tok = CharTokenizer.from_text("cab cab")
        self.assertEqual(tok.decode(tok.encode("cab")), "cab")
        with self.assertRaisesRegex(ValueError, "unknown"):
            tok.encode("z")

    def test_forward_and_generation(self):
        tokens = torch.randint(0, 10, (2, 8))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 10))
        self.assertTrue(torch.isfinite(loss))
        generated = self.model.generate(tokens[:, :2], 3)
        self.assertEqual(tuple(generated.shape), (2, 5))

    def test_future_tokens_cannot_change_past_logits(self):
        base = torch.tensor([[1, 2, 3, 4, 5]])
        altered = torch.tensor([[1, 2, 3, 8, 9]])
        with torch.no_grad():
            a, _ = self.model(base)
            b, _ = self.model(altered)
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-6))

    def test_short_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            split_corpus("abc" * 5, CharTokenizer.from_text("abc"), 8)

    def test_invalid_head_divisibility(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(vocab_size=10, n_embd=15, n_head=4)


class TrainingTests(unittest.TestCase):
    def test_trains_two_and_loads_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = main([
                "--minutes", "1", "--steps", "2", "--batch-size", "2",
                "--block-size", "16", "--embed-dim", "16", "--heads", "2",
                "--layers", "1", "--eval-batches", "1", "--sample-tokens", "4",
                "--device", "cpu", "--out-dir", tmp,
            ])
            self.assertEqual(len(summary["models"]), 2)
            for name in ("fields", "stories"):
                folder = Path(tmp) / name
                self.assertTrue((folder / "checkpoint.pt").is_file())
                self.assertTrue((folder / "sample.txt").is_file())
                metrics = json.loads((folder / "metrics.json").read_text())
                self.assertEqual(metrics["steps_completed"], 2)
                model, tokenizer, start = load_model(folder / "checkpoint.pt")
                self.assertIn(start, tokenizer.stoi)
                self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
            self.assertTrue((Path(tmp) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()

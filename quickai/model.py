"""A small causal, character-level Transformer and reversible tokenizer."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 48
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 2 or self.block_size < 2:
            raise ValueError("vocab_size and block_size must be >= 2")
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_layer <= 0:
            raise ValueError("embedding, heads and layers must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(chars) != len(set(chars)) or not chars:
            raise ValueError("Vocabulary must contain distinct characters")
        self.chars = list(chars)
        self.stoi = {char: i for i, char in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text).difference(self.stoi))
        if unknown:
            raise ValueError(f"Prompt includes unknown characters: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.chars[token] for token in tokens)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_width = width // self.n_head

        def split_heads(tensor):
            return tensor.reshape(batch, length, self.n_head, head_width).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v),
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        _, length = tokens.shape
        if not 1 <= length <= self.config.block_size:
            raise ValueError("sequence length must fit within the context window")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, count: int, temperature: float = 1.0):
        if count < 0 or temperature <= 0:
            raise ValueError("count must be nonnegative and temperature positive")
        for _ in range(count):
            logits, _ = self(tokens[:, -self.config.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probs, num_samples=1)), dim=1)
        return tokens

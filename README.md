# QuickAI

Train **two independent, ultra-small language models** from scratch with one command. QuickAI is a practical PyTorch learning experiment: a causal character-level Transformer trained on original field/science notes and another trained on original miniature stories. Both runs share a **20-minute wall-clock budget**, but typically finish much sooner at the default step limit on capable hardware. No API keys or external datasets are needed.

These models are **not general-purpose chatbots**. Their small parameter counts and tiny corpora make them pattern learners, prone to overfitting and repetition. Training speed is hardware-dependent; no guaranteed sub-minute CPU performance is claimed.

## Quick start

Use Python 3.10+ and install PyTorch (a CPU build is sufficient; for CUDA use the installation guide at https://pytorch.org/get-started/locally/).

```bash
python -m pip install -r requirements.txt
python -m quickai.train --minutes 20 --steps 300
```

The script selects CUDA, Apple MPS, or CPU automatically, trains `fields` then `stories`, and writes:

```
runs/
  summary.json
  fields/
    checkpoint.pt
    metrics.json
    sample.txt
  stories/
    checkpoint.pt
    metrics.json
    sample.txt
```

`metrics.json` includes initial/final training and holdout validation cross-entropy, iteration count, elapsed time, parameter count, configuration, dataset SHA-256, and seed. The 90/10 token split is contiguous. Validation is a small diagnostic rather than a robust quality benchmark.

Inspect the output or generate a new sample using a character that exists in that model's corpus:

```bash
python -m quickai.generate runs/fields/checkpoint.pt --prompt "A field" --tokens 150
python -m quickai.generate runs/stories/checkpoint.pt --prompt "The" --tokens 150
```

Unknown prompt characters raise a clear error. Checkpoints contain model weights and vocabulary, loaded with `weights_only=True`; do not load untrusted artifacts from unrelated sources.

## Run small, run larger

For a very fast smoke run, including on CPU:

```bash
python -m quickai.train --minutes 2 --steps 25 --batch-size 4 --block-size 32 --out-dir runs/smoke
```

For a larger experimental run (may reach its time budget):

```bash
python -m quickai.train --minutes 20 --steps 2000 --embed-dim 128 --layers 4 --heads 4 --out-dir runs/large
```

The time limit is shared, with up to half the *remaining* time assigned to each model. It is a **soft limit** checked between training iterations, not a real-time hard stop. The run can take slightly longer for initialization, evaluation, text sampling and checkpoint writes. A partial run is explicitly reported by `steps_completed`; completing `--steps` early is expected. Neither model is initialized from a pretrained model.

Use your own UTF-8 text files (each file must be long enough for the chosen context window and validation split):

```bash
python -m quickai.train --fields-data path/to/science.txt --stories-data path/to/fiction.txt
```

Datasets in `data/` were written specifically for this project and may be reused under the repository's Apache-2.0 license. Each model learns a **separate** character vocabulary and uses a distinct random seed. The architecture uses causal scaled-dot-product self-attention, residual Transformer blocks, AdamW, gradient clipping, held-out evaluation, and temperature-based sampling.

## Tests and reproducibility

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs the tests, performs a short two-model CPU training run, and uploads the `runs/` artifacts for inspection (7-day retention). Full-length training requires invoking the CLI on your machine or a cloud runner with adequate resources. Checkpoints and generated output are intentionally excluded from Git to keep the source repository small.

To reproduce a run, record the commit SHA, CLI arguments, Python/PyTorch versions, device, dataset hashes, and saved metrics. Seeds improve repeatability but CUDA/other kernels may still differ across platforms.

A measured CPU reference run is recorded in [benchmarks/local-cpu-2026-11-13.json](benchmarks/local-cpu-2026-11-13.json). Those values are hardware-specific, not a speed guarantee.

## License

Apache License 2.0. See [LICENSE](LICENSE).

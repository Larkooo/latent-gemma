# Latent Gemma

Research implementation of continuous-activation reasoning with Gemma. A learned
bridge feeds transformer hidden states back as input embeddings for a configurable
number of steps, then the model generates an answer. Latent steps do not sample
tokens or project states into vocabulary logits.

**Status: experiments in progress.** Correctness checks pass on small Gemma 3 and
Gemma 4 architectures. Accuracy and speed improvements have not been established.
This project does not reproduce Astra's undisclosed architecture.

## Scope

- MLX backend for Apple Silicon, with LoRA adaptation of existing model weights.
- Gemma 3 text and Gemma 4 text inference/training interfaces.
- Gemma 4 per-layer embeddings: continuous projection with zero token-table
  contribution on latent positions, so hidden states are not discretized by
  the upstream nearest-token fallback. Normal token positions are unchanged.
- Direct-answer, explicit chain-of-thought, and continuous-state experiments.
- Deterministic diagnostic datasets, public GSM8K data preparation, separate
  training/validation/test splits, prediction logs, synchronized GPU timing,
  uncertainty estimates, and activation ablations.

This is embedding recurrence through the full transformer, rather than recurrent
reuse of an internal subset of layers. Each latent position still costs computation.
See the [experiment protocol](docs/protocol.md).

## Install

Python 3.12+ and Apple Silicon are required for the current backend.

```sh
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## Reproduce a diagnostic experiment

Use a directory outside this checkout for model weights and experiment runs.

```sh
.venv/bin/python scripts/download_model.py \
  mlx-community/gemma-4-e2b-it-4bit ../work/models/gemma4 \
  --revision 238767527555cb75a05732a84dff5d6ba0dd6809
.venv/bin/latent-gemma data --output ../work/data/diagnostics
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --output ../work/runs/warmup --modes direct cot --steps 400 \
  --lora-layers 6 --rank 16 --batch-size 4 --learning-rate 0.00002
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --adapter ../work/runs/warmup/best --output ../work/runs/latent \
  --modes direct cot latent --latent-steps 4 --steps 600 --batch-size 4 \
  --learning-rate 0.00002
.venv/bin/latent-gemma evaluate \
  --model ../work/models/gemma4 --adapter ../work/runs/latent/best \
  --data ../work/data/diagnostics/validation.jsonl --mode latent \
  --latent-steps 4 --max-tokens 16 --output ../work/runs/latent-validation.jsonl
```

Evaluate `--mode direct` and `--mode cot` as matched controls. Use
`--ablation zero`, `--ablation shuffle`, and `--ablation repeat` with latent mode.
Use `--mode native --max-tokens 512` without an adapter for Gemma 4's native
thinking baseline. `--mode plain --max-tokens 512` uses the ordinary chat prompt
with native thinking disabled and no forced assistant prefix. This distinguishes
released-model behavior from the explicit formats used for fine-tuning. Report
truncation counts and decoding budgets for every comparison.
`auto` computation uses float32 for Gemma 4 because recurrent
training exposed nonfinite gradients with the original bfloat16 computation;
the integer weight storage remains quantized. Use the same computation dtype
for latency comparisons. See the [experiment log](docs/experiment-log.md) for
the numerical and token-alignment failures that were corrected during development.
Fix configuration choices on validation data before evaluating the test set.
An increase in output token efficiency alone is not evidence of lower latency.

## Staged compression

`hybrid` mode runs the latent loop, then generates the remaining text reasoning
and answer. During training, `--reasoning-steps-to-drop` removes initial reasoning
steps from the supervised continuation. It does not insert those steps into the
prompt. For example, a first curriculum stage can replace one reasoning step with
two latent positions:

```sh
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --adapter ../work/runs/warmup/best --output ../work/runs/stage1 \
  --modes cot hybrid --latent-steps 2 --reasoning-steps-to-drop 1 \
  --steps 400 --batch-size 4 --learning-rate 0.00002
.venv/bin/latent-gemma evaluate \
  --model ../work/models/gemma4 --adapter ../work/runs/stage1/best \
  --data ../work/data/diagnostics/validation.jsonl --mode hybrid \
  --latent-steps 2 --max-tokens 96 --output ../work/runs/stage1-validation.jsonl
```

Each new training invocation resets Adam's state and records the source adapter.
Arithmetic steps are the generated equations, link steps are graph edges, and
GSM8K steps are nonempty lines in its worked solution. When every step is removed,
the hybrid target contains only the answer delimiter and answer. Inference always
uses only the question, with a fixed latent count; no gold reasoning is supplied.
This follows the staged-compression idea in [Coconut](https://arxiv.org/html/2412.06769v2),
with different model adaptation and boundary handling. Its accuracy benefits in
this implementation still require measurement.

## Public benchmark data

```sh
.venv/bin/python scripts/prepare_gsm8k.py ../work/data/gsm8k
```

GSM8K validation examples come only from the original training split. Original
test examples remain test examples. Neither synthetic training nor small GSM8K
samples establish general-purpose model quality.

## Artifacts and licensing

Runs store adapter-only checkpoints, configuration, losses, and (for new runs)
source snapshots and dependency provenance. Base model weights are not included.
Gemma 3 checkpoints retain their Gemma terms; Gemma 4 is Apache-2.0. The code
license below does not relicense model weights or datasets.

Method references: [Coconut](https://github.com/facebookresearch/coconut),
[recurrent-depth research](https://arxiv.org/abs/2502.05171), and
[MLX LM](https://github.com/ml-explore/mlx-lm).

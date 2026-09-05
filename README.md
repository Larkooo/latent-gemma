# Latent Gemma

Research implementation of continuous-activation reasoning with Gemma. A learned
bridge feeds transformer hidden states back as input embeddings for a configurable
number of steps, then the model generates an answer. Latent steps do not sample
tokens or project states into vocabulary logits.

**Status: the latest hybrid pilot scored 96/100; performance checks are in progress.**
On 100 diagnostic validation examples, two continuous feedback steps followed by
a fixed text boundary and remaining text reasoning scored 96/100, versus 99/100
for the warmup text-reasoning reference. This is a hybrid of latent and text
reasoning. A separately trained shortened-text control scored 66/100, while
retaining 99/100 in full text mode. The [matched-training comparison](reports/matched-short-text-control/README.md)
supports useful latent computation on this pilot. Repeated latency measurements
and public-benchmark checks remain necessary before claiming a broader performance gain.
See the [fixed-boundary report](reports/curriculum-boundary-stage1/README.md) for
the complete accuracy matrix, raw predictions, and limitations.

Earlier recipes are preserved: [direct compression](reports/pilot-v2/README.md)
scored 80/100 and the [initial staged recipe](reports/curriculum-stage1/README.md)
scored 62/100. Their reports include the failed results and timing variability
that motivated interleaved measurements.
Correctness checks pass on small Gemma 3 and Gemma 4 architectures.
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

On macOS, Python ignores `.pth` files carrying the filesystem hidden flag. If an
editable install unexpectedly cannot be imported, inspect the virtualenv's
editable `.pth` file with `ls -lO` and clear that flag with `chflags nohidden` on
that file, or launch with `PYTHONPATH=/absolute/path/to/this/repo/src`. This issue
interrupted an experiment queue; setting its source path explicitly fixed it.

## Reproduce the fixed-boundary pilot

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
  --adapter ../work/runs/warmup/best --output ../work/runs/stage1 \
  --modes cot hybrid --latent-steps 2 --reasoning-steps-to-drop 1 \
  --hybrid-boundary reasoning --steps 400 --batch-size 4 \
  --learning-rate 0.00002
.venv/bin/latent-gemma evaluate \
  --model ../work/models/gemma4 --adapter ../work/runs/stage1/best \
  --data ../work/data/diagnostics/validation.jsonl --mode hybrid \
  --latent-steps 2 --max-tokens 96 --limit 100 \
  --output ../work/runs/stage1-validation.jsonl
```

The defaults use seed 42 and select a checkpoint every 100 updates. The recorded
run selected stage-one step 300. Exact results can vary with hardware and library
versions; the report records the environment and source hashes.
Evaluate `--mode cot` and `--mode hybrid --latent-steps 0` as inference controls.
Use `--ablation zero`, `--ablation shuffle`, and `--ablation repeat` with hybrid mode.
To test whether shortened text training works without latent positions, train
a separate control from the same warmup checkpoint:

```sh
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --adapter ../work/runs/warmup/best --output ../work/runs/short-text-control \
  --modes cot hybrid --latent-steps 0 --reasoning-steps-to-drop 1 \
  --hybrid-boundary reasoning --steps 400 --batch-size 4 \
  --learning-rate 0.00002
```

This matches the candidate's data, maximum updates, and shortened text targets;
training FLOPs are not equal because latent positions add computation.
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
New evaluations also record `end_to_end_latency_s`, from prompt formatting through
final decoding and answer extraction on a warm model. Compare that field with
`scripts/compare_runs.py --latency-field end_to_end_latency_s`; both prediction
files must contain it. Historical `latency_s` excludes prompt preparation and final
decoding. Neither measurement includes model loading or external serving overhead.

For timing claims, use repeated measurements with randomized, counterbalanced
condition order, loading both checkpoints before measurement:

```sh
.venv/bin/python scripts/benchmark_pair.py \
  --model ../work/models/gemma4 --adapter ../work/runs/stage1/best \
  --baseline-adapter ../work/runs/warmup/best \
  --data ../work/data/diagnostics/validation.jsonl \
  --output ../work/runs/paired-timing --candidate-mode hybrid \
  --latent-steps 2 --limit 100 --repeats 3 --max-tokens 96
```

The baseline defaults to text CoT. Each condition starts from a fresh attention
cache. Raw measurements are retained, and each question contributes once to the
accuracy comparison, using its median timing across repeats. Unexpected output
changes between repeats stop aggregation for investigation.
Comparisons report paired bootstrap intervals for speed ratios and the fraction
of questions answered faster. These intervals describe variation across questions;
they do not cover drift between hardware sessions.
Use `--baseline-adapter PATH --baseline-mode hybrid` to compare against a
separately trained zero-latent shortened-text control. Both adapters must use the
same base checkpoint and computation dtype. They are loaded before measurement;
this option requires memory for both model instances.

An experimental `--decode-strategy pipelined` evaluation option schedules the
next text token before reading the current token on the host. It can overlap
CPU graph construction with GPU execution. A request ending before its token cap
performs one unused continuation step; that work is included in time and position
counts, and is recorded in `prefetched_text_positions` and `vocabulary_projections`.
It is a latency optimization to measure, not a reduction in computation.
The default remains `serial`. To compare methods using the same pipelined decoder,
pass `--baseline-decode pipelined --candidate-decode pipelined` to
`scripts/benchmark_pair.py`. Token equivalence is tested on small Gemma models;
verification and timing on the real checkpoint remain pending.

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
  --hybrid-boundary reasoning \
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
with different model adaptation and boundary handling. The pilot outperformed the
tested shortened-text training control; broader transfer remains unestablished.

An optional `--hybrid-boundary reasoning` forces a fixed `\nReasoning: ` prefix
after the latent loop, before generating the remaining reasoning. Its training
tokens match generation exactly and are masked from the loss. This tests whether
a familiar text transition helps decoding from continuous states; it does not
supply an intermediate answer. The default is `none`. Evaluation inherits the
checkpoint's recorded boundary unless explicitly overridden. Boundary tokens
are included in timing and position counts. This option is experimental.

## Expanding the trainable layers

Gemma 4 E2B's last 20 layers reuse attention keys/values from earlier layers.
Adapting only the last six layers therefore leaves every key/value writer frozen.
To test whether adapting earlier layers helps, expand a saved adapter before
continuing training:

```sh
.venv/bin/python scripts/expand_adapter.py \
  --model ../work/models/gemma4 --adapter ../work/runs/warmup/best \
  --output ../work/runs/warmup-all-layers --lora-layers 35
```

This preserves existing weights and adds zero-output LoRA branches to earlier
layers. It verifies text and latent logits on a fixed probe before saving. The
same projection targets are used, including value projections where present;
key projections remain frozen. It increases training capacity and adapter cost,
and does not reduce the number of executed transformer layers. Any accuracy
benefit must be established through further training and evaluation.

## Public benchmark data

```sh
.venv/bin/python scripts/prepare_gsm8k.py ../work/data/gsm8k
```

GSM8K validation examples come only from the original training split. Original
test examples remain test examples. Neither synthetic training nor small GSM8K
samples establish general-purpose model quality.

Numeric answers use exact decimal-value comparison: `4.00` and `4` are equal,
while `4.000000000000001` and `4` are different. Link answers remain exact labels.
Rows record their scoring policy. To compare historical literal-string scores
with current results, rescore both files using `scripts/rescore.py SOURCE OUTPUT`.
It preserves the originals, records changed scores and source hashes, and saves
the scoring code with the new records. Comparisons reject mixed scoring policies.

## Artifacts and licensing

Runs store adapter-only checkpoints, configuration, losses, and (for new runs)
source snapshots and dependency provenance. Base model weights are not included.
Gemma 3 checkpoints retain their Gemma terms; Gemma 4 is Apache-2.0. The code
license below does not relicense model weights or datasets.

Method references: [Coconut](https://github.com/facebookresearch/coconut),
[recurrent-depth research](https://arxiv.org/abs/2502.05171), and
[MLX LM](https://github.com/ml-explore/mlx-lm).

# Reproduce the pilot

Run these commands from the repository root after the [setup](../README.md#setup).
Model weights and outputs go in a sibling `work/` directory. Use a new output
path for each experiment.

## Model and data

```sh
.venv/bin/python scripts/download_model.py \
  mlx-community/gemma-4-e2b-it-4bit ../work/models/gemma4 \
  --revision 238767527555cb75a05732a84dff5d6ba0dd6809
.venv/bin/latent-gemma data --output ../work/data/diagnostics
```

The dataset contains 6,000 training, 400 validation, 600 test, and 400 OOD
examples, using seed 20260905. Arithmetic and link traversal are balanced within
each split. The generator records hashes and excludes overlapping semantic IDs.

## Train

First train the text/direct-answer reference:

```sh
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --output ../work/runs/warmup --modes direct cot --steps 400 \
  --lora-layers 6 --rank 16 --batch-size 4 --learning-rate 0.00002
```

Then replace the first annotated reasoning step with two latent positions:

```sh
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --adapter ../work/runs/warmup/best --output ../work/runs/stage1 \
  --modes cot hybrid --latent-steps 2 --reasoning-steps-to-drop 1 \
  --hybrid-boundary reasoning --steps 400 --batch-size 4 \
  --learning-rate 0.00002
```

Defaults use seed 42 and evaluate checkpoint loss every 100 updates. The recorded
pilot selected warmup step 400 and hybrid step 300. Each training invocation
resets Adam's state. The fixed `\nReasoning: ` transition is masked from loss;
only the remaining reasoning and answer are supervised.

Gemma 4 defaults to float32 computation because recurrent training produced
nonfinite gradients in bfloat16. Integer weight storage remains quantized.

## Measure accuracy and latency together

Run this comparison directly after training:

```sh
.venv/bin/python scripts/benchmark_pair.py \
  --model ../work/models/gemma4 --adapter ../work/runs/stage1/best \
  --baseline-adapter ../work/runs/warmup/best \
  --data ../work/data/diagnostics/validation.jsonl \
  --output ../work/runs/stage1-comparison --candidate-mode hybrid \
  --latent-steps 2 --limit 100 --repeats 3 --max-tokens 96
```

Both checkpoints are loaded before measurement. Conditions alternate in
randomized, counterbalanced order, with fresh caches on every request. Outputs:

- `result.json` — accuracy, latency summaries, paired intervals, and configuration.
- `baseline.jsonl` and `candidate.jsonl` — one aggregate per question.
- `measurements.jsonl` — every timed request, token IDs, and output text.
- A neighboring `.provenance/` directory — source snapshots and input hashes.

Each question counts once for accuracy and uses its median time across repeats.
Changed outputs across repeats stop aggregation. Fix the recipe on validation
before substituting a test or OOD split.

## Controls

Individual evaluations use `latent-gemma evaluate` with model, adapter, data, and
output paths. Useful conditions are:

| Condition | Options |
|---|---|
| Full text reasoning | `--mode cot` |
| Same adapter, no latent positions | `--mode hybrid --latent-steps 0` |
| Feedback ablations | `--mode hybrid --latent-steps 2 --ablation zero`, `shuffle`, or `repeat` |
| Native Gemma thinking | `--mode native --max-tokens 1024`, without an adapter |
| Ordinary chat | `--mode plain --max-tokens 1024`, without an adapter |

Evaluation inherits the adapter's text boundary unless overridden. Report caps
and truncations alongside scores.

Train the shortened-text control from the same warmup reference:

```sh
.venv/bin/latent-gemma train \
  --model ../work/models/gemma4 --data ../work/data/diagnostics \
  --adapter ../work/runs/warmup/best --output ../work/runs/short-text-control \
  --modes cot hybrid --latent-steps 0 --reasoning-steps-to-drop 1 \
  --hybrid-boundary reasoning --steps 400 --batch-size 4 \
  --learning-rate 0.00002
```

Compare it with `benchmark_pair.py --baseline-adapter PATH --baseline-mode hybrid`.
This matches the data, maximum updates, and shortened targets. Training FLOPs
differ because latent positions add work.

## GSM8K and historical scores

```sh
.venv/bin/python scripts/prepare_gsm8k.py ../work/data/gsm8k
```

Preparation uses a pinned dataset revision. Validation comes from the original
training split; the original test split remains separate.

Numeric tasks use exact decimal-value equality: `4.00` equals `4`, while
`4.000000000000001` does not. Link labels require exact matches. To compare an old
literal-string score with current results, rescore both files:

```sh
.venv/bin/python scripts/rescore.py SOURCE.jsonl RESCORED.jsonl
```

Rescoring preserves originals, records changed scores and hashes, and saves the
scoring implementation. Comparisons reject mixed policies.

## Experimental options

`--decode-strategy pipelined` overlaps CPU graph construction with GPU execution
by scheduling a token ahead. Requests ending before the cap perform one unused
continuation; its time, position, and vocabulary projection are counted. For a
paired comparison, set both `--baseline-decode pipelined` and
`--candidate-decode pipelined`. Serial decoding is the default. The pilot report
does not measure a pipelined-decoder benefit.

To extend LoRA into earlier layers while preserving an existing adapter:

```sh
.venv/bin/python scripts/expand_adapter.py \
  --model ../work/models/gemma4 --adapter ../work/runs/warmup/best \
  --output ../work/runs/warmup-all-layers --lora-layers 35
```

Expansion adds zero-output branches and verifies logits before saving. It
increases training capacity; it does not reduce backbone depth or establish an
accuracy improvement without further training.

## Import troubleshooting

On macOS, a hidden filesystem flag on an editable-install `.pth` file can prevent
imports. Inspect it with `ls -lO` and clear the flag with `chflags nohidden`, or run
commands with `PYTHONPATH=/absolute/path/to/latent-gemma/src`.

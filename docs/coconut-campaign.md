# GSM8K curriculum campaign

This campaign tests whether recurrent feedback earns its accuracy/latency cost
against learned pause inputs, a shortened-text curriculum, and continued text
reasoning. It is a Gemma 4/LoRA adaptation of
[Coconut](https://arxiv.org/abs/2412.06769), not a replication of its published
GPT-2 results. No new performance result is established until the campaign
finishes and its artifacts pass the report audit.

## Training recipe

The default dataset has 6,973 official GSM8K training questions, 500 validation
questions held out from the original training split, and all 1,319 official test
questions. The dataset revision and file hashes are recorded. Unlike Coconut's
GPT-2 math experiment, this does not use its roughly 385,000 augmented examples.

Each of three seeds (42, 43, 44) starts from the pinned 4-bit Gemma 4 E2B model
and trains a three-epoch full-text warmup. Four arms then start from the same
per-seed warmup checkpoint:

| Arm | Stage 1 | Stage 2 | Stage 3 | Final stage |
|---|---|---|---|---|
| Recurrent feedback | 2 positions, remove 1 step | 4 positions, remove 2 steps | 6 positions, remove 3 steps | 6 positions, remove all reasoning |
| Learned pause | Same positions and removed steps as feedback | Same | Same | Same |
| Shortened text | 0 positions, remove 1 step | 0 positions, remove 2 steps | 0 positions, remove 3 steps | 0 positions, remove all reasoning |
| Full text CoT | 0 positions, keep all reasoning | Same | Same | Same |

Stages 1–3 each receive one complete shuffled epoch; the final stage receives
three. The optimizer resets at each stage for every arm, including CoT.
The default campaign totals 564,813 training example visits. Equal example and
update budgets do not imply equal training FLOPs or wall time.

The schedule uses two latent positions per removed reasoning step, with a fixed
position count even when a solution has fewer annotated steps. The last stage
removes every reasoning step, rather than leaving the tail of long solutions.
These choices follow the paper's staged compression procedure. The longer
warmup follows its larger-model appendix rather than the GPT-2 schedule.

LoRA rank 16 trains all 35 transformer layers, including Gemma 4's shared-cache
writers; the old diagnostic adapter trained only the last six. Gradients flow
through the recurrent states and functional caches. The frozen backbone uses
4-bit weights with float32 continuous computation. Effective batch size is
eight, implemented with one-example microbatches and token-weighted gradient
accumulation. AdamW uses learning rate 2e-5, no weight decay, and gradient norm
clipping at 1. A 16 GiB MLX allocation limit bounds memory use.

The pause control learns one question-independent embedding, shared across its
positions, and freezes the unused feedback bridge. It still attends to the
question and writes contextual states to the cache. This is stronger than
zeroing feedback only at inference. All arms retain the existing fixed text
boundary; there are no newly learned vocabulary boundary tokens. These are
explicit differences from the original Coconut implementation.

## Selection and evaluation

- All arms and seeds use the same fixed 128-question validation subset, selected
  by a hash of question ID. Validation runs after each epoch. Accuracy selects
  checkpoints, with validation loss breaking ties. Raw validation predictions
  and per-question losses are saved and hashed for every epoch. Training curves
  remain in the report; the epoch budget does not establish convergence.
- Stage transitions use the last checkpoint. Compressed arms select within
  their final stage. CoT can select its best checkpoint across the warmup and
  all continuation stages, so extra training cannot erase its best validation
  result.
- All checkpoint selections are frozen before any test evaluation. Every arm
  is evaluated on all 1,319 test questions with greedy pipelined decoding and the
  same 384-token cap. Incorrect and truncated answers remain in the results.
- Latency uses a separately frozen random subset of 128 test questions, three
  repeats per method, counterbalanced paired order, and synchronized warm
  requests. Feedback is compared separately with each control. This produces
  6,912 timed requests across three seeds. Model loading is excluded.
  Both sides use pipelined text decoding to overlap host scheduling and GPU
  execution; speed claims therefore do not depend on an unnecessarily serial
  text decoder. Any prefetched continuation after a stop token is synchronized
  and counted in request time and work counters. Validation uses the serial
  implementation of the same greedy decoding policy.
- Report both mean and median latency reductions, per-seed accuracy, truncation
  counts, text tokens, and nominal transformer positions. Positions are not a
  FLOPs or energy measurement. Report paired accuracy uncertainty by resampling
  both seeds and shared question IDs; three seeds give limited precision.

The final audit checks test coverage, scoring, checkpoint and artifact hashes,
completed training budgets, repeat agreement, counterbalancing, and per-question
latency medians. Timed outputs must match the corresponding full-test outputs.
`summary.json` and `report.md` are produced only after those checks pass.

A feedback advantage over learned pause would support the value of the
question-dependent feedback input in this setup. It would not, by itself,
establish general reasoning or the internal algorithm the model learned.

## Run and recover

From the repository, with its dependencies installed:

```sh
PYTHONPATH=src python scripts/run_campaign.py create \
  --model /absolute/path/to/gemma-4-e2b-it-4bit \
  --data /absolute/path/to/gsm8k \
  --output /absolute/path/to/campaign

# Run the frozen source, including after an interruption.
python /absolute/path/to/campaign/code/scripts/run_campaign.py run \
  /absolute/path/to/campaign

python /absolute/path/to/campaign/code/scripts/run_campaign.py status \
  /absolute/path/to/campaign
```

`create` snapshots source and data and hashes model/tokenizer files before
training. `run` verifies those inputs and the Python/package versions. Jobs run
sequentially; a shared lock prevents overlapping campaigns under the same run
directory, including workers left alive after a coordinator interruption.
Unrelated applications are outside that lock.

Each stage saves model and optimizer state every 50 updates and at epoch ends.
Only best and last checkpoints are retained. A deterministic per-update random
seed allows replay after an interruption. Partial evaluation attempts and
interrupted metric logs are archived before a retry; completed results are
reused. The runner stops on failed training, changed inputs, nonfinite gradients,
or inconsistent benchmark outputs rather than changing the recipe mid-run.

On macOS, `caffeinate -i` can wrap the frozen runner to prevent idle sleep for
the duration of the campaign. Closing the lid or shutting down may still
interrupt it. Resume with the same command and frozen campaign directory.

# Experiment log

All entries below are exploratory validation work. Final test sets have not yet
been evaluated. Failed experiments remain part of the record.

## Baseline and implementation checks

- Hardware: Apple M5, 32 GB unified memory, macOS 26.4.1.
- Runtime: Python 3.12.13, MLX 0.32.2, MLX LM 0.31.3.
- Initial Gemma 3 270M checkpoint:
  `mlx-community/gemma-3-270m-it-bf16`, revision
  `c806ef3a4ed971bd75aaee3346e0fef808512f03`.
- Gemma 4 checkpoint: `mlx-community/gemma-4-e2b-it-4bit`, revision
  `238767527555cb75a05732a84dff5d6ba0dd6809`.
- Synthetic dataset: 6,000 train, 400 validation, 600 test, 400 longer/larger-input
  OOD examples. Arithmetic and link traversal are balanced within each split.
  Exact semantic IDs do not overlap across any split.
- GSM8K was fetched at repository revision
  `3101c7d5072418e28b9008a6636bde82a006892c`. Its original training data supplies
  6,973 training and 500 validation examples; original test contains 1,319.
  No exact question-ID overlap was found across the resulting splits.

## Pilot results

1. Gemma 3 270M, 100 validation problems: direct answer 1/100, explicit CoT 9/100.
   These results include format failures and are evidence that this checkpoint
   is not a sufficient demonstration of useful reasoning in this setup.
2. Gemma 3 270M, 600 supervised steps on direct/CoT examples: completed, best
   mean validation token loss 1.1304. This loss is invalid for comparing modes
   because of the answer-prefix masking bug described below. This early pilot
   predates automatic source snapshots and is excluded from final results.
3. Gemma 4 E2B, native thinking, original quantized checkpoint computation dtype,
   40 validation problems: **40/40**, median latency 5.512 seconds, mean 186.125
   generated tokens, no truncation at the 512-token limit. This is a small,
   deliberately simple diagnostic sample, not evidence of general accuracy.
   Original predictions were re-scored after fixing an answer-parser bug: an
   occurrence of the string `Answer:` inside a quoted thought could consume the
   actual final answer on the same line. Generation text and timing are unchanged;
   originals and rescored files both remain in the run directory.

## Numerical failures and corrections

- An embeddings-only full-sequence test exposed in-place input scaling in the
  Gemma 3 implementation. The wrapper now preserves caller-owned embeddings.
- Float32 CPU comparisons isolate semantic cache equivalence from Metal's
  shape-dependent rounding. Real checkpoint inference is measured on the GPU.
- The first quantized Gemma 4 training smoke run failed at its first latent batch:
  loss was finite but gradients were NaN. A controlled replay of the same batches
  reproduced this with the checkpoint's bfloat16 computation. Float32 computation
  produced finite gradients on all six replayed updates. This establishes a
  precision-sensitive failure; it does not establish the specific kernel cause.
- `auto` computation now selects float32 for Gemma 4, leaving integer quantized
  weight storage intact. New evaluations record this setting. The native baseline
  above used original bfloat16 computation, so runtime comparisons must disclose
  the difference or include an additional matched float32 baseline.
- The trainer now checks loss and gradient norm before every optimizer update and
  records a failure rather than applying a nonfinite gradient.
- A real-tokenizer audit exposed another issue: concatenating `Answer: ` with a
  letter lets the tokenizer merge the space and letter. Computing the loss-mask
  length from the separately encoded delimiter therefore masked the first answer
  token. This affected all 3,000 link-training examples, and 3 GSM8K answers would
  also have been affected. Direct/latent training now concatenates the exact
  decoder prefix token IDs with the answer token IDs, supervising every answer
  token. CoT targets and all generation scoring are unchanged. A regression test
  covers this boundary for both direct and latent modes.
- The original Gemma 4 warmup and the subsequent K=4 run are excluded from valid
  training results because of this masking bug. The K=4 run was interrupted after
  its step-100 validation. Clean `gemma4-warmup-v2` and `gemma4-latent-k4-v2` runs
  restart from the base checkpoint with corrected labels and source snapshots.

## Corrected training and additional checks

- The real-tokenizer alignment audit passed for all 6,000 diagnostic and 6,973
  GSM8K training examples in both direct and latent modes: forced prefixes match
  inference, all answer tokens are supervised, and answer tokens round-trip.
- Re-running pinned GSM8K preparation reproduced every manifest field and split
  hash exactly.
- Gemma 4 cached feedback gradients agree with full-sequence recomputation to
  relative vector error below 1e-4 on the small float32 test architecture. Causal
  target alignment also passes with shared KV layers. The suite now has 42 tests.
- `gemma4-warmup-v2` completed 400 updates, batch size 4, learning rate 2e-5,
  alternating direct/CoT modes. Its selected checkpoint is step 400, with mean
  validation token losses 0.22037 (direct) and 0.00166 (CoT). Elapsed training plus
  periodic validation was 356.90 seconds. These are losses, not accuracy results.
- `gemma4-latent-k4-v2` continues from that checkpoint for 600 planned updates,
  alternating direct/CoT/latent modes. Gradients remain finite but can have very
  large norms before clipping. Accuracy and activation ablations are queued.
- A separate staged-compression path is implemented: `hybrid` decoding runs
  continuous steps and then generates remaining text reasoning. Training can
  remove initial annotated reasoning steps while retaining the rest as targets.
  This enables a curriculum; it does not establish a performance improvement.

## Remaining experiments

Validation generation, activation ablations, training-budget controls, staged
compression if needed, a public reasoning benchmark, and a frozen final test
evaluation remain necessary before claiming success.

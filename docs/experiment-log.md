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
   mean validation token loss 1.1304. Generation accuracy still requires evaluation.
   This early pilot predates automatic source snapshots.
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

## Running next

Gemma 4 direct/CoT warmup, followed by mixed direct/CoT/latent training at K=4.
Next steps are validation generation, activation ablations, training-budget
controls, a public reasoning benchmark, and a frozen final test evaluation.

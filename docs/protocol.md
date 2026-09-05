# Experiment protocol

## Objective

Build a working continuous-activation reasoning implementation for Gemma, measure
accuracy and performance, and prepare a technically justified ecosystem proposal.
This is not an implementation or reproduction of Astra's undisclosed architecture.

The initial target was to match the text-reasoning reference's answer accuracy
while reducing time to the completed answer. After observing the fixed-boundary
pilot, 96/100 against the 99/100 text reference was accepted as a useful quality
milestone, so this candidate proceeds to timing and mechanism checks. This is an
explicit post-result acceptance decision, not a prespecified statistical
equivalence result or a universal tolerance for future benchmarks. Continue
improving accuracy while establishing whether the tradeoff is useful.

Lower compute is a separate target: wall-clock speed, memory use, and operation
count are not interchangeable. Report paired uncertainty and the observed
quality/latency tradeoff. A successful small pilot alone is insufficient for
claims of broad reasoning quality.

## Mechanism

Prompt -> transformer hidden state -> learned normalization/projection bridge ->
continuous input embedding -> transformer (repeat K times) -> answer tokens.
Every latent step has a position in the attention cache. The full backbone runs
on that position. No token is sampled and no vocabulary projection is performed
inside the latent loop. This is a Coconut-inspired adaptation, not a claim to
reproduce its exact training curriculum or results.

## Measurements and controls

- Establish pretrained, direct-answer SFT, text-CoT SFT, and latent-answer models.
- Use identical base weights and disjoint train/validation/test data.
- Report all experiment configurations, including failed experiments.
- Select settings/checkpoints using validation only. Freeze settings before test.
- Measure exact-answer accuracy, per-task counts, uncertainty, median/p95 latency,
  generated text tokens, latent positions, and peak memory. Synchronize GPU timing.
- New runs measure both model latency (`latency_s`) and warm request latency
  (`end_to_end_latency_s`). The latter includes prompt formatting/tokenization,
  prompt prefill, latent computation, forced prefixes, generation through stop or
  cap, detokenization, and answer extraction. Model loading, warmup, and external
  serving/network overhead are excluded. Never substitute model latency when an
  old run lacks the end-to-end field; rerun both sides for that comparison.
- For speed claims, interleave conditions on the same questions with randomized,
  counterbalanced order and repeated measurements. Count each question once for
  accuracy, and retain raw trials. Check equivalent decoding paths as a timing
  control; single sequential runs can drift with machine load and thermal state.
- Compare accuracy versus measured latency, not just token counts. CoT has more
  text positions; latent positions still consume transformer computation.
- Ablate latent states (zero, corrupted features, repeated first state). This
  checks whether input-dependent recurrent activations matter beyond extra slots.
  These controls alter the explicit feedback input; they retain attention caches.
  A zero input can still produce problem-dependent cached activations by attending
  to the prompt. A null feedback result therefore does not prove that all latent
  computation is irrelevant. Matched training without latent positions remains
  necessary to distinguish useful latent memory from shorter text supervision.
- Test cache equivalence, causal target alignment, gradient flow through feedback,
  checkpoint round trips, and the zero-step path against original model outputs.
- Evaluate a public reasoning benchmark in addition to generated diagnostic tasks.
- Do not claim broad reasoning improvements from small or synthetic benchmarks.

## Initial decisions (before training)

Start with Gemma 3 270M on Apple Silicon to debug training and caching, then move
to a larger checkpoint when the pipeline works. MLX supplies the existing model
implementation and LoRA; custom code supplies only feedback, data, and experiments.
The initial pilot used a two-percentage-point accuracy tolerance against the
matched text-CoT baseline. The subsequently clarified objective above targets
matching accuracy and improving latency; the earlier tolerance remains recorded
for interpreting that historical pilot. Report uncertainty and failures even if
the target is not reached. An implementation that works but does not retain
useful accuracy is not a successful end state.

Retain legible reasoning as an explicit alternative mode. Text explanations and
activation probes are not assumed faithful or sufficient for oversight. Monitorability
is a separate research question; this small experiment cannot establish safety.

## Cost and compression hypotheses

A useful approximation for a warm request is prompt-processing time plus
`K * cost_per_latent_step + T * cost_per_text_step`, plus text preparation and
decoding overhead. The two step costs need not be equal. Avoiding the vocabulary
projection saves some work, but a full-transformer latent step remains expensive.
Nominal transformer positions are a diagnostic counter, not a measured FLOP count;
lazy execution can eliminate computations that do not affect the output.

First test gradual replacement of text steps and a fixed transition back into
text. Sweep latent count and retained text using validation, preserving a strong
text reference and feedback ablations. A shortened text solution that works
equally well without the latent state is not evidence of useful latent reasoning.

If full-transformer recurrence becomes the limiting cost after accuracy is
retained, investigate a smaller recurrent block and the adaptation it requires.
Reusing only part of Gemma would change its computation and cache semantics;
simply skipping pretrained layers does not establish a valid recurrent model.
[Published recurrent-depth work](https://arxiv.org/abs/2502.05171) demonstrates
training a shared internal block, with more recurrence spending more computation.
It does not establish Astra's architecture or guarantee a faster Gemma conversion.

## Upstream path

First produce code, reproducible runs, and a measured result. Prepare a proposal
for a narrow experimental example or reusable embedding/hidden-state interface.
Maintainer discussion and a library PR do not modify Google's released weights.
Do not submit a speculative architecture change without evidence. Google requires
a CLA for contributions; do not sign agreements on another person's behalf.

## References

- https://github.com/facebookresearch/coconut
- https://arxiv.org/abs/2412.06769
- https://arxiv.org/abs/2502.05171
- https://github.com/google-deepmind/gemma
- https://github.com/ml-explore/mlx-lm
- https://arxiv.org/abs/2507.11473

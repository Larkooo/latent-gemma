# Experiment protocol

## Objective

Build a working continuous-activation reasoning implementation for Gemma, measure
accuracy and performance, and prepare a technically justified ecosystem proposal.
This is not an implementation or reproduction of Astra's undisclosed architecture.

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
- Compare accuracy versus measured latency, not just token counts. CoT has more
  text positions; latent positions still consume transformer computation.
- Ablate latent states (zero, corrupted features, repeated first state). This
  checks whether input-dependent recurrent activations matter beyond extra slots.
- Test cache equivalence, causal target alignment, gradient flow through feedback,
  checkpoint round trips, and the zero-step path against original model outputs.
- Evaluate a public reasoning benchmark in addition to generated diagnostic tasks.
- Do not claim broad reasoning improvements from small or synthetic benchmarks.

## Initial decisions (before training)

Start with Gemma 3 270M on Apple Silicon to debug training and caching, then move
to a larger checkpoint when the pipeline works. MLX supplies the existing model
implementation and LoRA; custom code supplies only feedback, data, and experiments.
The initial performance target is within two percentage points of the matched
text-CoT baseline, with lower measured answer latency. Report uncertainty and
failures even if this target is not reached. An implementation that works but
does not retain useful accuracy is not a successful end state.

Retain legible reasoning as an explicit alternative mode. Text explanations and
activation probes are not assumed faithful or sufficient for oversight. Monitorability
is a separate research question; this small experiment cannot establish safety.

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

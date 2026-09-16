# September 2026 runtime feasibility checkpoint

Status: experimental baseline, not a mathematically structured reasoning system.
Published 2026-09-16. These are new runtime checks, separate from the historical
September 5 diagnostic study linked in the repository README.

## Question and method

Can the existing Gemma/MLX implementation run inference and a finite gradient
update, then save and reload its trainable arrays? This tests execution, not
whether latent feedback improves mathematical reasoning.

The source baseline is commit
[`5cbc558b6ee089e9b0156c268db67e9a4a3d0dbc`](https://github.com/Larkooo/latent-gemma/tree/5cbc558b6ee089e9b0156c268db67e9a4a3d0dbc).
The backbone is [`mlx-community/gemma-4-e2b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit)
at revision `238767527555cb75a05732a84dff5d6ba0dd6809`. Weights are not
redistributed here; upstream [Gemma terms](https://ai.google.dev/gemma/terms)
continue to apply.

## Observed results

Two deliberately simple development prompts, `7 + 8` and `9 × 3`, produced
`15` and `27` with the base model and no latent positions. The process took
4.480450 seconds including startup/loading, with a recorded MLX peak of
3,105,419,728 bytes. This is not a benchmark or a reliable latency estimate.

A separate fresh-adapter check used exactly one training example (`4 + 5`,
answer `9`) and one development diagnostic example (`6 + 7`, answer `13`).
Configuration: one latent step, LoRA rank 2/scale 16 on one layer, bridge rank 4,
seed 20260915, float32 computation, batch size 1, AdamW learning rate 0.0001,
weight decay 0, gradient-norm clipping at 1.0. The original `encode_example`
and `token_loss` implementation supplied the targets and teacher-forced loss.

- Exactly one optimizer update completed with finite recorded losses and all
  nine gradient arrays finite.
- Five arrays changed, comprising 20,481 parameter elements.
- All nine trainable arrays were bitwise equal after saving and reloading.
- Process wall time: 5.909529 seconds, including startup/loading and reload.
- Recorded MLX peak: 6,012,661,788 bytes. This is not total system memory.

The tiny teacher-forced losses changed from 11.188037 to 10.806749 on the
training example, and from 7.581203 to 7.460211 on the diagnostic example.
These are descriptive smoke-check observations, not evidence of accuracy,
generalization, convergence, or a causal benefit from latent reasoning.

## Reproduction and evidence boundary

The pinned baseline's [setup](../../README.md#setup) and
[reproduction guide](../../docs/reproduce.md) document the existing public model,
data, training and evaluation interfaces. They require an appropriate Apple
Silicon/MLX environment. The full historical training recipe is not the same
as this one-update smoke check; this note does not claim an exact standalone
replay of the new check is published.

This publication is a reviewed summary of recorded execution. The temporary
execution supervisor is not included: subsequent review found monitoring and
resource-accounting defects, so it is not approved as a reusable harness.
No new inference or training was run to publish this note. Original source,
model and historical evaluation artifacts remain unchanged.

## What remains unestablished

The existing mechanism feeds an unconstrained learned continuous state back
into a transformer. It does not implement explicit mathematical objects,
proof-preserving transitions, or an algebraic state representation. Those are
a distinct research direction, not a capability demonstrated here.

There is no new controlled multi-step comparison, recovered historical adapter
replication, or reasoning-quality improvement in this checkpoint. Do not combine
these development observations with the old held-out study or present the
historical results as outcomes of this new run.

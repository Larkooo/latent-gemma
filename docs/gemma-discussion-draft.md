# Gemma discussion draft

Local draft only; nothing has been posted. Publish the reproducible repository
before posting and add its public URL to the evidence paragraph. Refresh the
independent-evaluation status when those results finish.

Suggested category: **Show and tell**, with a question about contribution scope.

## Suggested title

Gemma 4 hybrid continuous-state reasoning: measured accuracy/latency pilot

## Proposed post

We have implemented a Gemma 4 E2B hybrid continuous-state reasoning experiment
using MLX and LoRA. Two learned continuous feedback positions replace an initial
text reasoning step; the model then generates the remaining reasoning and answer.
The feedback loop does not sample tokens or map activations to their nearest
vocabulary entry. For latent positions, the implementation preserves Gemma 4's
continuous per-layer embedding branch and zeros the token-indexed contribution.

On 100 diagnostic arithmetic/link validation questions:

| Method | Correct answers | Median completed-answer latency |
|---|---:|---:|
| Warmup text-CoT reference | 99/100 | 1.339 s |
| Two-step hybrid | 96/100 | 1.103 s |

This is 17.6% lower median latency, measured with three repeats per method and
question (600 requests), randomized and counterbalanced condition order, fresh
caches, identical base checkpoint/dtype, and the same serial decoder. The
baseline/candidate latency ratio was 1.214 with a paired-question bootstrap 95%
interval of [1.105, 1.376]. The speed benefit was concentrated in arithmetic;
link-question latencies were approximately equal.

A separately trained control with the same shortened text targets and maximum
update budget, but no latent positions, scored 66/100. Its full-text mode retained
99/100. This supports a useful role for the latent positions under this recipe;
training FLOPs are not matched, and only one seed has been tested.

These are exploratory diagnostic validation results from one Apple Silicon
session, with validation also used for checkpoint selection. The experiment is
hybrid latent/text reasoning, not fully language-free reasoning. It does not
establish lower total FLOPs, general reasoning superiority, or another lab's
architecture. Initial untrained GSM8K transfer scored 24/40 for both the hybrid
and warmup-CoT reference, versus 28/40 for the candidate adapter in full-text mode.
An independent combined accuracy/latency check on 600 diagnostic test and 400
harder OOD questions is running with the recipe frozen.

The accompanying repository contains the implementation, pinned checkpoint and
adapter settings, training instructions, raw predictions, all 600 timing trials,
mechanism controls, failed experiments, and source snapshots. This builds on
continuous-thought research such as Coconut; we are not claiming the underlying
idea is new.

Would a reproducible community example be useful to the Gemma ecosystem? We would
also appreciate guidance on whether a supported continuous-embedding input path
belongs in this library. At the inspected Gemma commit, the token-based forward
method already supports hidden-state returns; the proposed extension would reuse
the existing transformer internals while specifying embedding scaling, per-layer
inputs without token IDs, positions, masks, and cache behavior. Our current MLX
implementation would need a separate JAX/Flax port and tests before becoming a
library PR.

## Local evidence to link after publication

- [Combined accuracy and latency](../reports/boundary-accuracy-latency/README.md)
- [Matched shortened-text training control](../reports/matched-short-text-control/README.md)
- [Logical inference work](../reports/inference-work-audit/README.md)
- [Detailed contribution proposal](upstream-proposal.md)

## Contribution route

- [Gemma Discussions](https://github.com/google-deepmind/gemma/discussions)
- [Contribution requirements](https://github.com/google-deepmind/gemma/blob/main/CONTRIBUTING.md)
- [Google's earlier showcase of a Gemma/Coconut community project](https://developers.googleblog.com/en/unlock-global-communication-gemma-projects/)

The showcase is evidence of interest in community continuous-reasoning work;
it is not an assurance of acceptance for this experiment or a library change.

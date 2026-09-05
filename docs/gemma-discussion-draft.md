# Gemma 4 hybrid continuous-state reasoning

We have implemented a Gemma 4 E2B experiment that replaces an initial text
reasoning step with two learned continuous feedback positions, then generates
the remaining reasoning and answer. The loop uses no token sampling or nearest
vocabulary lookup. It preserves Gemma 4's continuous per-layer embedding branch
and zeros the token-indexed contribution at latent positions.

On 100 diagnostic arithmetic and link-traversal validation questions:

| Method | Correct | Median completed-answer latency |
|---|---:|---:|
| Text-CoT reference | 99/100 | 1.339 s |
| Two-step hybrid | 96/100 | 1.103 s |

The hybrid reduced median latency by 17.6%. Both methods were measured three
times per question with randomized, counterbalanced order, fresh caches, the
same base checkpoint/dtype, and serial decoding. The speed ratio was 1.214 with
a paired-question 95% interval of [1.105, 1.376]. Arithmetic supplied most of the
benefit; link-question latencies were approximately equal.

A separately trained control with the same shortened text targets and maximum
update budget, but no latent positions, scored 66/100 while retaining 99/100 in
full-text mode. Training FLOPs were not matched.

This is a single-seed, single-machine diagnostic pilot. Validation also informed
checkpoint selection. The model retains some text reasoning, lower total FLOPs
have not been measured, and initial GSM8K transfer is mixed. Independent test and
OOD evaluation is running. The approach builds on
[Coconut](https://arxiv.org/abs/2412.06769).

The repository includes [code and reproduction instructions](../README.md),
[raw timing measurements](../reports/boundary-accuracy-latency/README.md),
[matched training controls](../reports/matched-short-text-control/README.md),
and [earlier unsuccessful recipes](experiment-log.md).

Would a reproducible community example be useful? We would also appreciate
feedback on a [supported continuous-embedding input path](upstream-proposal.md).
The existing token-based interface already returns hidden states; the extension
would specify embedding scaling, per-layer inputs without token IDs, positions,
masks, and caches. A library PR would need a separate JAX/Flax implementation and
tests.

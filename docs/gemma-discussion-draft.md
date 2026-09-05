# Gemma 4 hybrid continuous-state reasoning

Hello everyone! I recently successfully implemented an experiment on Gemma 4 E2B
that replaces an initial text reasoning step with two learned continuous feedback
positions, then generates the remaining reasoning and final answer. The loop uses
no token sampling or nearest vocabulary lookup, preserves Gemma 4's continuous
per-layer embedding branch, and zeros the token-indexed contribution at latent
positions!

Here are some benchmarks I ran on 100 diagnostic arithmetic and link-traversal
validation questions:

| Method | Correct | Median completed-answer latency |
|---|---:|---:|
| Text-CoT reference | 99/100 | 1.339 s |
| Two-step hybrid | 96/100 | 1.103 s |

The hybrid answered 96 out of 100 questions correctly and reduced median
completed-answer latency by 17.6%! I measured both methods three times per
question, randomized and balanced which method ran first, and used fresh caches,
the same base checkpoint and dtype, and serial decoding. The speed ratio was
1.214, with a paired-question 95% interval of [1.105, 1.376]. Most of the speedup
came from arithmetic; the link-traversal questions took about the same time.

I also trained a control with the same shortened text targets and maximum update
budget, but without latent positions. It scored 66/100 while still reaching
99/100 in full-text mode. I haven't matched training FLOPs between the two, so
this comparison doesn't establish equal training compute.

I'm encouraged by these results, though this is still a small diagnostic pilot
with one seed on one machine. I used validation results to help select the
checkpoint, and the model still generates some text reasoning. I haven't measured
whether it uses fewer total FLOPs, and the initial GSM8K transfer results are
mixed. Independent test and out-of-distribution evaluations are running. This
experiment builds on [Coconut](https://arxiv.org/abs/2412.06769).

I've included [code and reproduction instructions](../README.md),
[raw timing measurements](../reports/boundary-accuracy-latency/README.md),
[matched training controls](../reports/matched-short-text-control/README.md),
and [earlier experiments that didn't work as well](experiment-log.md) in the
repository.

I'd love to hear what you think! Would a reproducible community example like this
be useful? I'd also appreciate feedback on a
[supported continuous-embedding input path](upstream-proposal.md). The existing
token-based interface already returns hidden states; the proposed extension would
define how embedding scaling, per-layer inputs without token IDs, positions,
masks, and caches should work. To turn that into a library PR, I'd need to add a
separate JAX/Flax implementation and tests.

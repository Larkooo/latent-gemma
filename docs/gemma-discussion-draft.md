# Gemma 4 hybrid continuous-state reasoning

Hello everyone! I recently successfully implemented an experiment on Gemma 4 E2B
that replaces an initial text reasoning step with two learned continuous feedback
positions, then generates the remaining reasoning and final answer. The loop uses
no token sampling or nearest vocabulary lookup, preserves Gemma 4's continuous
per-layer embedding branch, and zeros the token-indexed contribution at latent
positions!

I used two latent steps as a starting point. The count is configurable, but this
checkpoint was trained with two; I haven't yet systematically compared other
counts for accuracy and latency.

Here are some benchmarks I ran on 100 diagnostic arithmetic and link-traversal
validation questions:

| Method | Correct | Median completed-answer latency |
|---|---:|---:|
| Text-CoT reference | 99/100 | 1.339 s |
| Two-step hybrid | 96/100 | 1.103 s |

The hybrid answered 96 out of 100 questions correctly and reduced median
completed-answer latency by 17.6%! To reduce timing noise and order effects, I
measured both methods three times per question and randomized and balanced which
one ran first. Both used fresh caches, the same base checkpoint and computation
dtype, and the same serial decoder. Most of the speedup came from arithmetic;
the link-traversal questions took about the same time.

I also trained a separate control on the same shortened text targets, but without
latent steps. It scored 66/100, mostly struggling with arithmetic, while still
reaching 99/100 with full written reasoning. This suggests the latent steps help
preserve accuracy when shortening the reasoning. The maximum number of training
updates was matched, though the total training compute was not.

The most promising part for me is getting the final answer sooner, which could
be useful when running Gemma locally on smaller devices. These timings came from
an Apple M5 laptop with 32 GB of memory. I haven't tested bigger Gemma models yet,
but I'd expect the balance between the cost of latent steps and the time saved
by generating less text to change with model size and hardware. I'd be interested
to see whether that leads to a similar or larger speedup!

For now, these are pilot results from one training seed on one machine, and the
validation questions also informed checkpoint selection. Lower total inference
compute still needs to be measured.

I've included [code and reproduction instructions](../README.md),
[raw timing measurements](../reports/boundary-accuracy-latency/README.md),
[matched training controls](../reports/matched-short-text-control/README.md),
and [earlier experiments that didn't work as well](experiment-log.md) in the
repository.

I'd love to hear what you guys think! Would a reproducible community example like this
be useful?

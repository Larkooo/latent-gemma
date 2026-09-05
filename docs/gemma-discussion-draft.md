# Gemma 4 hybrid continuous-state reasoning

Hello everyone! I recently successfully implemented an experiment on Gemma 4 E2B
that replaces an initial text reasoning step with two learned continuous feedback
positions, then generates the remaining reasoning and final answer. The loop uses
no token sampling or nearest vocabulary lookup, preserves Gemma 4's continuous
per-layer embedding branch, and zeros the token-indexed contribution at latent
positions!

I used two latent steps as a starting point. The count is configurable, but this
checkpoint was trained with two. I also have experimented with one additional latent step with inference taking 5.3% longer by median latency but no apparent gain in accuracy.

Here are some benchmarks I ran on 600 held-out diagnostic arithmetic and
link-traversal test questions:

| Method | Correct | Median completed-answer latency | Mean completed-answer latency |
|---|---:|---:|---:|
| Text-CoT reference | 598/600 (99.67%) | 1.031 s | 1.195 s |
| Two-step hybrid | 583/600 (97.17%) | 0.922 s | 0.994 s |

Hybrid approach answered 583 out of 600 questions correctly and reduced median
completed-answer latency by 10.6%! Mean latency fell by 16.8%. That's a
2.5 percentage-point accuracy loss for a faster final answer.

I measured both methods twice per question, with fresh caches and randomized,
balanced order, for 2,400 timed requests. The checkpoints and decoding settings
were fixed before this test, and these questions were not used for checkpoint
selection. All questions are included, including incorrect answers. Most of the
speedup came from arithmetic; link-traversal timings were about the same.

I also trained a separate control on the same shortened text targets, but without
latent steps. It scored 66/100 on the earlier validation sample. This suggests
the latent steps help preserve accuracy when shortening the reasoning.

The most promising part for me about this is the gain in performance, which is especially useful for Gemma's smaller models that are especially used in less capable devices.

Benchmark came from my Macbook Pro M5 laptop with 32 GB of memory. I haven't tested bigger Gemma models yet but I'd be curious to see how it holds up, especially by properly tuning the number of latent steps. I'd expect around the same gain in saved latency.

I've included [code and reproduction instructions](https://github.com/Larkooo/latent-gemma/blob/main/README.md),
[full test results and raw timing measurements](https://github.com/Larkooo/latent-gemma/blob/main/reports/independent-test/README.md),
[the two-versus-three-step comparison](https://github.com/Larkooo/latent-gemma/blob/main/reports/latent-step-count/README.md),
[matched training controls](https://github.com/Larkooo/latent-gemma/blob/main/reports/matched-short-text-control/README.md),
and [earlier experiments that didn't work as well](https://github.com/Larkooo/latent-gemma/blob/main/docs/experiment-log.md) in the
repository.

I'd love to hear what you guys think!

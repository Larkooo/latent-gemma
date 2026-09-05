# Equivalent decoding paths measured the same speed

**The repeated timing sanity check measured a speed ratio of 1.0003,
with a paired bootstrap 95% interval of [0.9733, 1.0101].** This is consistent
with equal speed. It supports using this measurement procedure; it is not evidence
that latent reasoning is faster.

The same loaded no-boundary stage-one checkpoint answered the first ten diagnostic
validation questions in CoT mode and in hybrid mode with zero latent steps. These
paths generated identical text on all ten questions and both scored 10/10. Each
condition was measured four times per question, with randomized, counterbalanced
order and a fresh attention cache. The raw trace contains 80 measured requests.

| Measurement | CoT | Zero-latent hybrid |
| --- | ---: | ---: |
| Median warm request, seconds | 1.260948 | 1.260585 |

Each question contributes once to accuracy and uses the median of its four timing
trials. Warm request timing includes prompt formatting/tokenization, model
computation, generation, final decoding, and answer extraction. It excludes model
loading, warmup, and external serving overhead. Bootstrap intervals describe
question sampling, not uncertainty from different hardware sessions.

The previous sequential evaluation of these paths produced substantially different
timings despite identical outputs. This small repeated check addresses that concern
for this session. The subsequent
[primary comparison](../boundary-accuracy-latency/README.md) uses the same
measurement procedure.

All raw hashes and question/repeat counts were verified. No timed requests were
rerun or discarded.

Pinned model, adapter hash, dependency versions, decoding conditions, source
snapshots, raw trials, and question aggregates accompany this report. Absolute
workspace prefixes in exported metadata are replaced, and the original result
hash is retained. Base and adapter weights are not included.

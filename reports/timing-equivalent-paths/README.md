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
for the present session; larger candidate comparisons remain necessary.

After all measurements and the result file were saved, the outer queue's log reader
failed on a numeric line of pretty-printed JSON. All raw hashes and question/repeat
counts were verified. The logger and benchmark console output were corrected, and
the deferred primary candidate comparison was queued after the current serial jobs.
No timed requests from this check were rerun or discarded.

Pinned model, adapter hash, dependency versions, decoding conditions, source
snapshots, raw trials, and question aggregates accompany this report. Absolute
workspace prefixes in exported metadata are replaced, and the original result
hash is retained. Base and adapter weights are not included.

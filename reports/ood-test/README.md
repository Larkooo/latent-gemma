# Out-of-distribution test: the tradeoff does not hold on larger operands

On all 400 OOD questions, the two-step hybrid scored
**360/400 (90.00%)** versus
**389/400 (97.25%)** for the text-CoT reference.
Median completed-answer latency fell by **19.4%**,
from **0.7748 s** to **0.6242 s**.

| Task | Questions | Text-CoT accuracy | Hybrid accuracy | Text-CoT median, s | Hybrid median, s |
|---|---:|---:|---:|---:|---:|
| overall | 400 | 97.25% | 90.00% | 0.7748 | 0.6242 |
| arithmetic | 200 | 95.00% | 80.50% | 0.9827 | 0.7610 |
| links | 200 | 99.50% | 99.50% | 0.4987 | 0.5047 |

The paired accuracy difference was -7.25%, with a 95% interval of
[-10.25%, -4.25%]. The baseline/hybrid
latency ratio was 1.241, with an interval of [1.186, 1.289].
Truncated outputs: text 0, hybrid 11.

## What this establishes

The OOD split uses arithmetic operands from 20 to 50 instead of 1 to 20, and
link chains of five or six hops instead of two to four. Link accuracy was
unchanged. Arithmetic accuracy dropped by 14.5 percentage points, which is far
outside the 2.5-point loss on the in-distribution test.

Of the 40 hybrid arithmetic errors, 26 emitted a wrong value for the silently
computed first product, 11 abandoned the trained format and fell back into the
base model's step-by-step chat style until the 96-token cap, 2 wrote the first
step in text instead of skipping it, and one was a link error shared with the
reference. Two latent positions do not carry a two-digit by two-digit product
reliably, and when the latent state is off-distribution the adapter's format
control fails.

This result was measured in the same frozen run as the independent test, with
the same candidate, reference, decoder, and token cap. It was not published in
the first version of this repository. It is the strongest evidence in the
repository that the two-step hybrid is a compression of memorized step
patterns, not a general shortcut.

## Measurement and verification

Both methods were measured twice per question with randomized, counterbalanced
order and fresh caches, producing 1600 timed requests. Repeated token sequences
and correctness agreed for every question. Each question counts once for
accuracy and uses its median latency across repeats. All questions are
included, including incorrect and truncated outputs.

Timing scope, hardware, and software match the
[independent test report](../independent-test/README.md). Absolute latencies in
this run are inflated by concurrent load on the machine; see the
[session calibration note](../../docs/protocol.md#session-calibration). The
paired ratio within the run is unaffected by that constant.

The report preserves raw trials, predictions, all OOD questions, the frozen
plan, adapter settings, and artifact hashes. Source snapshots are identical to
the independent test report and are not duplicated here. Base and adapter
weights are not included.

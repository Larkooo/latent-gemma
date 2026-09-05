# Independent diagnostic test

On all 600 held-out test questions, the two-step hybrid scored
**583/600 (97.17%)** versus
**598/600 (99.67%)** for the text-CoT reference.
Median completed-answer latency fell by **10.6%**, from
**1.0315 s** to
**0.9224 s**.

Mean completed-answer latency fell by **16.8%**, from **1.1945 s** to
**0.9938 s**. Both statistics use the same per-question aggregated timings and
include all 600 questions. These mean values were added as a descriptive
cross-check after the median comparison; the primary timing metric remains the
ratio of median latencies.

| Task | Questions | Text-CoT accuracy | Hybrid accuracy | Text-CoT median, s | Hybrid median, s |
|---|---:|---:|---:|---:|---:|
| overall | 600 | 99.67% | 97.17% | 1.0315 | 0.9224 |
| arithmetic | 300 | 100.00% | 94.67% | 1.8387 | 1.3086 |
| links | 300 | 99.33% | 99.67% | 0.8137 | 0.8101 |

The paired accuracy difference was -2.50%, with a
95% interval of [-3.83%, -1.33%].
The baseline/hybrid latency ratio was 1.118,
with an interval of [1.069, 1.152]. The hybrid answered faster on
70.0% of questions. Its p95 latency was
1.6147 s versus
2.1912 s for text CoT.
Truncated outputs: text 0, hybrid 1.

## What this establishes

The accuracy/latency tradeoff persisted on new diagnostic questions with the
candidate, reference, source, decoder, and token cap fixed before evaluation.
These test questions did not select the checkpoints or training recipe. The
overall median latency reduction is smaller than the 17.6% validation pilot.
Arithmetic supplies the speed benefit; link latencies are approximately equal.

An overall median is not an average of task-specific improvements. The pilot
contained 53 arithmetic and 47 link questions; this test contains 300 of each.
The timing distribution and task mix affect which questions lie near the median.
For comparison, mean latency fell by 17.6% in the pilot and 16.8% in this test.
These are different samples measured in different sessions, so the difference
cannot be attributed entirely to task mix.

This remains one training seed, one machine, and two procedural task families.
It is evidence of generalization to new questions from this diagnostic
distribution, not broad reasoning superiority. The harder 400-question OOD
evaluation is separate and not included here. The measured accuracy loss is
real; these results do not establish accuracy equivalence. Total FLOPs and energy
have not been measured.

## Measurement and verification

Both methods were measured twice per question with randomized, counterbalanced
order and fresh caches, producing 2400 timed requests. Each method ran first in
600 paired trials. Repeated token sequences and correctness agreed for every
question. Each question counts once for accuracy and uses its median latency
across repeats. All questions are included, including incorrect and truncated
outputs. Intervals resample paired questions; they do not account for hardware
variation between sessions.

Timing includes prompt formatting, tokenization, prefill, latent steps, forced
transition, generation, detokenization, and answer extraction. It excludes model
loading, warmup, and external serving. Both adapters were loaded before timing;
both use serial decoding and a 96-token cap. Hardware: Apple M5, 32 GB memory,
macOS 26.4.1. The three-step validation experiment started only after this test
finished, so its timed GPU workload did not overlap.

The data audit independently recomputed every answer across 7400 questions and
verified canonical IDs and zero overlap across train, validation, test, and OOD.
This verifies data integrity and separation, not benchmark representativeness.
The report preserves raw trials, predictions, all test questions, the frozen
plan, adapter settings, source snapshots, and artifact hashes.

Use the [reproduction instructions](../../docs/reproduce.md), substituting this
report's `test.jsonl`, `--limit 600`, and `--repeats 2` in the paired benchmark
command. Base and adapter weights are not included in this report.

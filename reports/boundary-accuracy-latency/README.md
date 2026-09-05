# Boundary hybrid: accuracy and completed-answer latency

The two-step hybrid scored **96/100** versus
**99/100** for the warmup text-CoT reference. Median warm
request latency was **1.1033 seconds** for the
hybrid and **1.3391 seconds** for text CoT.
The baseline/candidate latency ratio was **1.214**,
with a paired-question bootstrap 95% interval of **[1.105, 1.376]**.
A ratio above one favors the hybrid; below one favors text CoT.

| Task | Questions | Text accuracy | Hybrid accuracy | Text median, s | Hybrid median, s | Text/hybrid latency |
|---|---:|---:|---:|---:|---:|---:|
| overall | 100 | 99.0% | 96.0% | 1.3391 | 1.1033 | 1.214 |
| arithmetic | 53 | 98.1% | 92.5% | 1.6569 | 1.2315 | 1.346 |
| links | 47 | 100.0% | 100.0% | 0.8327 | 0.8141 | 1.023 |

Both methods ran on the same 100 diagnostic validation questions, with three
measurements per question and condition: 600 requests in total. Condition order
was randomized and counterbalanced, with each method first in 150 paired trials.
The same base checkpoint and computation dtype were used. Both adapters were
loaded before timing, every request received fresh attention caches, and both
methods used serial decoding with a 96-token cap. All three trials for each
question and condition produced identical token IDs and correctness.

Each question contributes once to accuracy and uses its median time across
repeats. Timing includes prompt formatting/tokenization, model computation,
latent steps, forced transition tokens, generation, detokenization, and answer
extraction. It excludes model loading, warmup, external serving, and network
overhead. All measured questions are included, including incorrect answers.
Truncated outputs: text 0, hybrid 0.

The paired accuracy difference was -3.0%, with a 95%
bootstrap interval of [-8.0%, +1.0%].
The hybrid answered faster on 74.0% of questions.
The median of per-question speed ratios was 1.213.
These paired intervals describe sampling questions from this diagnostic set;
they do not measure between-session hardware variation.

This is one training seed, exploratory validation that overlaps checkpoint
selection, and simple arithmetic/link tasks. It is not an independent test or a
broad reasoning benchmark. The 96/100 development target was adopted after
observing the results; it was not a prespecified equivalence margin.
The hybrid still generates some text reasoning. This timing experiment does not
measure FLOPs or energy; see the separate
[logical-work audit](../inference-work-audit/README.md).

Raw trials, per-question aggregates, all 100 input questions, adapter settings,
pinned model and source metadata, and inference source snapshots accompany this
report. The original result hash is retained; absolute workspace prefixes in
exported metadata are replaced. Model and adapter weight files are not included.

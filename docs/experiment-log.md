# Experiment results

The frozen recipe has completed its independent diagnostic test and
[harder OOD comparison](../reports/ood-test/README.md). OOD arithmetic accuracy
was 161/200 versus 190/200 for text reasoning; links were 199/200 for both.
Development results below use validation questions. The new
[GSM8K curriculum campaign](coconut-campaign.md) is a separate experiment.
Reports retain predictions, settings, source snapshots, and hashes, including
unsuccessful experiments.

## Independent test: accuracy and latency

The two-step candidate and text-CoT reference were fixed before evaluating all
600 test questions. These questions were not used for checkpoint selection.

| Method | Accuracy | Median completed-answer latency |
|---|---:|---:|
| Warmup text-CoT reference | 598/600 (99.67%) | 1.0315 s |
| Two-step hybrid | 583/600 (97.17%) | 0.9224 s |

Median latency was 10.6% lower, with a 2.5 percentage-point accuracy loss. The
paired accuracy interval was [−3.83, −1.33] points; the baseline/hybrid latency
ratio was 1.118, with an interval of [1.069, 1.152]. Two repeats per condition
produced 2400 timed requests. All repeated outputs agreed. The hybrid hit the
token cap on one question; the reference did not.
Mean completed-answer latency fell from 1.1945 s to 0.9938 s, a 16.8% reduction;
this is a descriptive cross-check using the same per-question timings.

Arithmetic accuracy was 284/300 versus 300/300, with 28.8% lower median latency.
Link accuracy was 299/300 versus 298/300, with approximately equal latency.
This supports the tradeoff on new questions from these two task families; it
does not establish broad reasoning quality or equivalent accuracy.

[Independent test and raw trials](../reports/independent-test/README.md)

## Validation pilot: accuracy and latency

The current pilot uses Gemma 4 E2B with two latent positions, a fixed
`\nReasoning: ` transition, and remaining text reasoning.

| Method | Accuracy | Median completed-answer latency |
|---|---:|---:|
| Warmup text-CoT reference | 99/100 | 1.3391 s |
| Two-step hybrid | 96/100 | 1.1033 s |

Three repeats per method and question produced 600 timed requests. All repeated
token sequences agreed and no output hit the 96-token cap. The hybrid's median
latency was 17.6% lower; the speed ratio was 1.2137 with a paired-question 95%
interval of [1.1053, 1.3764]. It answered faster on 74% of questions.

Arithmetic accounted for most of the benefit: its ratio of median latencies was
1.3455. The link-task ratio was 1.0229, with an interval spanning equal speed.
Accuracy differed by −3 percentage points, with a paired interval of [−8, +1].
These intervals cover question sampling within one hardware session.

[Full comparison and raw trials](../reports/boundary-accuracy-latency/README.md)

## Training variants

### Inference step count

On the existing two-step checkpoint, a direct two-versus-three-step comparison
scored 96/100 in both conditions. Three steps gained three answers and lost
three. Its median latency was 0.6506 s versus 0.6180 s, or 5.3% longer. The
two-step/three-step latency ratio was 0.950, with an interval of [0.898, 1.020].
This does not establish a clear overall timing difference or compare models
trained for each count. The same 100 validation questions and three repeats
produced 600 timed requests, with identical repeated outputs. All two-step
outputs also matched the original pilot. See the
[step-count report](../reports/latent-step-count/README.md).

### Training recipes

| Recipe | Latent steps | Accuracy | Evidence |
|---|---:|---:|---|
| Direct compression | 4 | 80/100 | [Report](../reports/pilot-v2/README.md) |
| Hybrid without a text transition | 2 | 62/100 | [Report](../reports/curriculum-stage1/README.md) |
| Hybrid with a fixed transition | 2 | 96/100 | [Report](../reports/curriculum-boundary-stage1/README.md) |
| Separately trained shortened-text control | 0 | 66/100 | [Report](../reports/matched-short-text-control/README.md) |

The fixed-transition candidate and shortened-text control started from the same
warmup checkpoint, used the same targets, data, seed, and maximum 400 updates,
and selected steps 300 and 400 respectively. Both retained 99/100 in full-text
mode. The hybrid solved 30 additional questions and lost none, with a paired
accuracy-difference interval of [21, 39] percentage points. Training FLOPs were
not matched.

For the fixed-transition candidate, zeroed feedback scored 93/100, reversed
features 90/100, repeated initial feedback 86/100, and zero latent positions
90/100. These controls retain the attention cache, so they do not erase every
problem-dependent latent state.

## Compression and work

On 53 arithmetic questions, the hybrid produced a correct answer with exactly
the intended shortened equation on 46, versus 16 for the trained control. Each
had three additional correct but noncanonical continuations.
[Continuation audit](../reports/arithmetic-compression-audit/README.md)

Against the same adapter's full-text mode, the hybrid emitted 29.1% fewer tokens.
Mean nominal transformer positions increased from 62.85 to 63.42 because of the
latent steps and transition. Vocabulary projections decreased, but these logical
counters do not measure total FLOPs or energy.
[Work audit](../reports/inference-work-audit/README.md)

## Initial GSM8K transfer

None of these adapters was trained on GSM8K. Forty validation questions were
scored with exact numeric equivalence:

| Condition | Correct | Output cap | Truncated |
|---|---:|---:|---:|
| Native pretrained thinking | 29/40 | 1,024 | 8 |
| Warmup text CoT | 24/40 | 768 | 3 |
| Candidate adapter, full text | 28/40 | 768 | 1 |
| Candidate adapter, hybrid | 24/40 | 768 | 3 |

Different inference formats and caps limit comparisons with native thinking.
This sample does not show a broad reasoning improvement. The separate
shortened-text control stopped after 32 questions; its incomplete run is excluded.

## Reproducibility

- Hardware: Apple M5, 32 GB memory, macOS 26.4.1.
- Runtime: Python 3.12.13, MLX 0.32.2, MLX LM 0.31.3.
- Base: `mlx-community/gemma-4-e2b-it-4bit`, revision
  `238767527555cb75a05732a84dff5d6ba0dd6809`.
- Diagnostics: 6,000 train / 400 validation / 600 test / 400 OOD; disjoint semantic IDs.
- GSM8K: revision `3101c7d5072418e28b9008a6636bde82a006892c`;
  6,973 train / 500 validation / 1,319 test.

Earlier runs affected by an answer-prefix loss-mask bug were excluded and
retrained. Numeric scoring was corrected to accept equivalent decimal forms;
original records remain preserved. Sequential timing drift motivated the
[repeated equivalent-path check](../reports/timing-equivalent-paths/README.md)
and the current interleaved comparison.

The independent run freezes the two-step candidate, reference, data, source,
serial decoder, and 96-token cap. It measures accuracy and latency together on all
600 test and 400 OOD questions, with two repeats per method. The complete test
result is above; the [OOD report](../reports/ood-test/README.md) records the harder split.
The [protocol](protocol.md) describes selection and scope;
[reproduction instructions](reproduce.md) provide the commands.

# Two versus three latent steps

The existing checkpoint scored **96/100 with two steps** and
**96/100 with three steps**. Median completed-answer latency
was **0.6180 s** and
**0.6506 s**, respectively.

Three steps took 5.3% longer by the ratio of median latencies and provided no net
accuracy gain. The overall timing interval includes equal speed, so this does
not establish a precise slowdown or an optimal step count.

| Task | Questions | Two-step accuracy | Three-step accuracy | Two-step median, s | Three-step median, s |
|---|---:|---:|---:|---:|---:|
| overall | 100 | 96.0% | 96.0% | 0.6180 | 0.6506 |
| arithmetic | 53 | 92.5% | 92.5% | 0.7464 | 0.7856 |
| links | 47 | 100.0% | 100.0% | 0.4993 | 0.5272 |

The two-step/three-step latency ratio was 0.950,
with a paired-question 95% interval of [0.898, 1.020]. A ratio
above one favors three steps. Three steps gained 3
correct answers and lost 3; the accuracy difference
was +0.0%, with an interval of
[-5.0%, +5.0%].
Truncated outputs: two steps 0, three steps 0.

## Measurement

Both conditions used the same loaded adapter, fresh caches, the same fixed text
transition, serial decoding, and a 96-token cap. Each of the 100 questions was
measured three times per condition, for 600 timed requests. Order was randomized
and counterbalanced. Repeated outputs were identical, and all two-step outputs
matched the original pilot exactly. Each question counts once for accuracy and
uses its median latency across repeats.

Latency covers prompt formatting through final answer extraction, including
prefill, latent steps, the transition, and generated text. It excludes model
loading, warmup, and external serving. Every question is included, including
incorrect answers. The preceding independent test split finished before this
comparison began; the OOD split resumed afterward. Timed GPU workloads did not
overlap. Hardware: Apple M5, 32 GB memory, macOS 26.4.1.

## Scope

This checkpoint was trained with two latent steps. Adding a third at inference
tests sensitivity to a changed step count; it does not test a model trained for
three steps. These are the first 100 diagnostic validation questions, including
the 32 used for checkpoint selection. Prior development used this sample.
Results therefore remain exploratory, and the intervals do not account for
selection bias or hardware variation between sessions. This experiment measures
latency and logical work counters, not total FLOPs or energy.

The report preserves all questions, raw trials, source snapshots, checkpoint
settings, the plan recorded before measurement, and the scheduling record.
Base model and adapter weights are not included.

## Reproduction

After the repository setup and checkpoint preparation in the
[reproduction guide](../../docs/reproduce.md), run from the repository root:

```sh
PYTHONPATH=src python reports/latent-step-count/reproduce.py \
  --model PATH_TO_BASE_MODEL --adapter PATH_TO_TWO_STEP_ADAPTER \
  --data reports/latent-step-count/validation-sample.jsonl \
  --output ../work/runs/latent-step-count-reproduction
```

This compares two and three steps on the same loaded checkpoint. Absolute
latencies should not be compared directly with the separate two-adapter
text-CoT/hybrid benchmark; use the paired comparison within each experiment.
`original-run.py` records the original local scheduling; `reproduce.py` runs the
comparison without depending on those processes.

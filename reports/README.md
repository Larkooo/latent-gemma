# Reports

Each report preserves the inputs, predictions, settings, and source snapshots
used for its conclusions. Historical files retain their original scoring and
measurement metadata; use the current result summaries when comparing methods.

## Current pilot

| Report | Result |
|---|---|
| [Independent diagnostic test](independent-test/README.md) | 583/600 versus 598/600; 10.6% lower median answer latency across 2400 requests. |
| [Validation accuracy and latency](boundary-accuracy-latency/README.md) | 96/100 versus 99/100; 17.6% lower median answer latency across 600 requests. |
| [Two versus three latent steps](latent-step-count/README.md) | Both 96/100; three-step median latency 5.3% higher, with an interval including equal speed. |
| [Matched shortened-text training](matched-short-text-control/README.md) | 96/100 with latent steps versus 66/100 without them. |
| [Feedback controls](curriculum-boundary-stage1/README.md) | Fixed-boundary recipe and activation ablations. |
| [Arithmetic continuations](arithmetic-compression-audit/README.md) | Correct, exactly shortened continuations on 46/53 versus 16/53 questions. |
| [Logical inference work](inference-work-audit/README.md) | 29.1% fewer generated tokens; total compute not measured. |

## Earlier experiments and measurement checks

| Report | Result |
|---|---|
| [Direct compression](pilot-v2/README.md) | Four latent positions scored 80/100. |
| [Hybrid without a transition](curriculum-stage1/README.md) | Two latent positions scored 62/100. |
| [Equivalent decoding paths](timing-equivalent-paths/README.md) | Repeated timing control was consistent with equal speed. |

The [results summary](../docs/experiment-log.md) also covers initial GSM8K transfer
and the independent evaluation in progress.

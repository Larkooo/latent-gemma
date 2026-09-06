# Stage-1 diagnostics: why the compressed curriculum lost accuracy

This note records a bounded follow-up to the cancelled GSM8K curriculum
campaign (`docs/coconut-campaign.md`). It uses the campaign's frozen data, its
seed-42 full-text warmup adapter, and the same 128 validation questions. Those
questions are now development data; the remaining 372 validation questions are
reserved for confirmation and were not used. The official test set was not
touched.

All accuracies are greedy serial decoding with a 384-token cap, scored with
`numeric-equivalence-v1`. Every prediction file is preserved under
`work/runs/claude-stage1-diagnostics` and `work/runs/claude-stage1`.

## Reproduction and inference ablations

The saved stage-1 adapter (2 latent positions, first reasoning step removed,
872 updates at 2e-5) reproduces its recorded 66/128 with all 128 generations
byte-identical. Ablating the recurrent input at inference on that adapter:

| Feedback at inference | Correct | Truncated | Generations identical to unablated |
|---|---:|---:|---:|
| As trained | 66/128 | 1 | 128 |
| Zeroed | 69/128 | 1 | 72 |
| Reversed (shuffle) | 64/128 | 2 | 70 |

The recurrent content is not used. The latent positions act as untrained pause
slots.

## Teacher-forced loss on the values that must be computed silently

`scripts/diagnose_carried_values.py` locates, in the supervised text, the first
mention of each number that the removed step computed and later text reuses
(120 of 128 questions have one). Mean loss over all supervised tokens hides
those few tokens.

| Adapter, protocol | Mean loss | Carried-value loss | Carried-value top-1 | Questions with every carried value top-1 |
|---|---:|---:|---:|---:|
| Stage 1, feedback | 0.544 | 0.723 | 0.806 | 78/120 |
| Stage 1, feedback zeroed | 0.546 | 0.735 | 0.803 | 77/120 |
| Stage 1, feedback reversed | 0.546 | 0.730 | 0.797 | 77/120 |
| Warmup (no compression training), 2 latent positions | 0.830 | 2.490 | 0.471 | 17/120 |
| Warmup, 0 latent positions | 0.826 | 2.578 | 0.472 | 16/120 |

Per question, the feedback-minus-zeroed difference in carried-value loss has
mean -0.013 and median -0.002. Stage-1 training raised silent value computation
from 47% to 81% top-1, entirely through paths that do not depend on the
recurrent input.

## Where the free-generation errors come from

`scripts/compression_error_analysis.py` classifies how each generation handles
the removed step's final computed value.

| Run | Value recomputed in text | Value used silently | Value absent | Correct when absent |
|---|---:|---:|---:|---:|
| Stage 1 feedback (66/128) | 28 (17 correct) | 55 (40 correct) | 40 | 6 |
| Shortened-text control (66/128) | 19 (8 correct) | 63 (42 correct) | 41 | 11 |

When the value appears, accuracy is close to the 76.6% full-text warmup. The
loss comes from the third of questions where the model never computes it.
Over-skipping is secondary: 29 of the feedback run's generations contain one
fewer reasoning line than the shortened target, and 20 of those are wrong.

## Matched control: zero latent positions, same shortened targets

Trained from the same warmup with identical data order, budget, and settings
(872 updates at 2e-5, one epoch, first step removed):

| Arm | Correct | Validation loss | Mean generated tokens |
|---|---:|---:|---:|
| Feedback, 2 latent positions | 66/128 | 0.544 | 69.2 |
| Shortened text, 0 latent positions | 66/128 | 0.547 | 69.8 |

Paired: 53 both correct, 13 only feedback, 13 only shortened text, 49 both
wrong. The two arms are different models at the same accuracy. At this budget
the latent positions and their feedback contribute nothing.

## Pending runs

Queued after this note was drafted, all from the same warmup and budget:

- Feedback at learning rate 1e-4, with the same ablation diagnostics.
- Feedback at 1e-4 with an auxiliary objective that decodes the removed step's
  result from the final latent state (`--value-aux-weight 0.5`).
- Feedback at 1e-4 with the carried-value tokens upweighted in the text loss
  (`--carried-value-weight 5`).
- Shortened-text and learned-pause controls at the same learning rate.

Results will be appended here with their prediction hashes.

## Interpretation so far

- Nothing indicates a defect in the recurrent path; the tests and diagnostics
  show it changes computation and receives gradients.
- The recipe never gave the model a reason to use it. The shortened task can be
  solved by attention over the question, and one epoch at 2e-5 with LoRA is
  roughly 40x fewer and weaker updates than the paper's GSM8K schedule
  (learning rate 1e-4, three epochs per stage, batch 32 on 385k examples).
- The paper itself reports Coconut below text reasoning on GSM8K (34.1% vs
  42.9%); its gains are on logical-search tasks. A small accuracy cost on GSM8K
  was never promised by the method.

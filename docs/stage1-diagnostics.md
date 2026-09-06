# Stage-1 diagnostics: why the compressed curriculum lost accuracy

This note records a bounded follow-up to the cancelled GSM8K curriculum
campaign (`docs/coconut-campaign.md`), run on September 6, 2026. It reuses the
campaign's frozen data, its seed-42 full-text warmup adapter, and the same 128
validation questions. Those questions are now development data. The remaining
372 validation questions are reserved for confirmation and were not used. The
official test set was not touched. Everything is one seed, so differences of a
few questions are within noise (the 95% interval at n=128 is about ±8 points).

All accuracies use greedy serial decoding with a 384-token cap, scored with
`numeric-equivalence-v1`. Prediction files are preserved under
`work/runs/claude-stage1-diagnostics` (ablations, teacher-forced diagnostics)
and `work/runs/claude-stage1` (new trainings); hashes are listed at the end.

## Summary table

Every row starts from the same warmup and removes the first reasoning step.
Trained rows use one epoch, batch 8, 872 updates.

| Run | Latent positions | Learning rate | Extra | Correct | Feedback zeroed at inference |
|---|---:|---:|---|---:|---:|
| Full-text warmup (reference) | 0 | 2e-5 | keeps all reasoning | 98/128 | n/a |
| Feedback stage 1 (campaign) | 2 | 2e-5 | | 66/128 | 69/128 |
| Shortened-text control | 0 | 2e-5 | | 66/128 | n/a |
| Feedback | 2 | 1e-4 | | 45/128 | 47/128 |
| Feedback + removed-value decoding | 2 | 2e-5 | `--value-aux-weight 0.5` | 56/128 | 60/128 |
| Feedback + carried-value weighting | 2 | 2e-5 | `--carried-value-weight 5` | stopped at update 220 | |
| Learned-pause control | 2 | 2e-5 | | not started | |

The last two rows were cancelled at the user's request.

## Reproduction and inference ablations

The saved stage-1 adapter reproduces its recorded 66/128 with all 128
generations byte-identical, so paired comparisons below are exact.

| Feedback at inference (stage-1 adapter) | Correct | Truncated | Generations identical to unablated |
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
| Stage 1 (2e-5), feedback | 0.544 | 0.723 | 0.806 | 78/120 |
| Stage 1 (2e-5), feedback zeroed | 0.546 | 0.735 | 0.803 | 77/120 |
| Stage 1 (2e-5), feedback reversed | 0.546 | 0.730 | 0.797 | 77/120 |
| Feedback 1e-4 | 0.583 | 0.727 | 0.773 | 71/120 |
| Feedback 1e-4, zeroed | 0.582 | 0.731 | 0.774 | 71/120 |
| Feedback + value decoding (2e-5) | 0.560 | 0.796 | 0.778 | 74/120 |
| Feedback + value decoding, zeroed | 0.565 | 0.799 | 0.780 | 76/120 |
| Warmup (no compression training), 2 latent positions | 0.830 | 2.490 | 0.471 | 17/120 |
| Warmup, 0 latent positions | 0.826 | 2.578 | 0.472 | 16/120 |

Per-question feedback-minus-zeroed differences in carried-value loss have mean
-0.013 (stage 1), -0.004 (1e-4), and -0.003 (value decoding). Stage-1 training
raised silent value computation from 47% to 81% top-1, entirely through paths
that do not depend on the recurrent input.

## Where the free-generation errors come from

`scripts/compression_error_analysis.py` classifies how each generation handles
the removed step's final computed value (123 questions have one).

| Run | Recomputed in text | Used silently | Absent | Correct when absent |
|---|---:|---:|---:|---:|
| Feedback stage 1 (66/128) | 28 (17 correct) | 55 (40 correct) | 40 | 6 |
| Shortened-text control (66/128) | 19 (8 correct) | 63 (42 correct) | 41 | 11 |
| Feedback 1e-4 (45/128) | 19 (8 correct) | 61 (29 correct) | 43 | 6 |
| Feedback + value decoding (56/128) | 20 (12 correct) | 59 (38 correct) | 44 | 4 |

When the value appears, accuracy is close to the 76.6% full-text warmup. The
loss comes from the third of questions where the model never computes it.
Over-skipping is secondary: 29 of the stage-1 generations contain one fewer
reasoning line than the shortened target, and 20 of those are wrong.

## Matched control: zero latent positions, same shortened targets

Trained from the same warmup with identical data order, budget, and settings:

| Arm | Correct | Validation loss | Mean generated tokens |
|---|---:|---:|---:|
| Feedback, 2 latent positions | 66/128 | 0.544 | 69.2 |
| Shortened text, 0 latent positions | 66/128 | 0.547 | 69.8 |

Paired: 53 both correct, 13 only feedback, 13 only shortened text, 49 both
wrong. The two arms are different models at the same accuracy. At this budget
the latent positions and their feedback contribute nothing.

## Learning rate 1e-4 is too high for this LoRA setup

Only the learning rate changed (2e-5 to 1e-4, the paper's value for full
fine-tuning of GPT-2). Training loss stayed above the 2e-5 curve throughout
(0.64 vs 0.50 over updates 201-300) with gradient-norm spikes up to 55. The
1e-4 model makes new arithmetic slips inside steps it still writes
("$200 - $198 = $12"), so the damage is general. Paired against the 2e-5 run:
34 both correct, 32 only 2e-5, 11 only 1e-4. Zeroing its feedback at inference
gives 47/128, so the higher rate did not make the model use the recurrent path
either.

## Giving the latent state an explicit reason to carry the value

`--value-aux-weight 0.5` adds a branch that decodes the removed step's result
(its digit tokens) from the final latent state on a cloned KV cache, so the
supervised text never sees those tokens. Per-example smoke tests showed this
branch sends roughly 400x more gradient into the bridge than the text loss.

The branch is learnable: its loss fell from about 4.5 nats per digit at
initialization to roughly 1.1 by the end of the epoch (inferred from the logged
total minus the text loss). It did not help the text: 56/128, with 20 questions
lost and 10 gained against the baseline, and the carried-value loss rose from
0.723 to 0.796. Zeroing the feedback at inference gives 60/128, and the
teacher-forced difference is again zero. The latent position learned to compute
the value from attention over the question, not from its recurrent input, and
the extra objective competed with the text objective rather than feeding it.

## Final conclusion

For Gemma 4 E2B with 4-bit weights and rank-16 LoRA on one laptop, removing
the first GSM8K reasoning step costs about 25 accuracy points (98 to 66 of 128)
at a one-epoch budget, and recurrent latent feedback does nothing to reduce
that cost:

- The trained model ignores the feedback input at 2e-5, at 1e-4, and with an
  auxiliary objective that directly rewards encoding the hidden value.
- Two latent positions tie the zero-position shortened-text control exactly.
- The one lever the paper relies on, much larger update budgets, hurts here:
  5x the learning rate degrades general arithmetic.

No useful accuracy/latency tradeoff from latent feedback was found, and the
planned paired latency benchmark was not run because there is no accuracy-
matched candidate to time. This is consistent with the paper's own GSM8K table
(Coconut 34.1% versus text CoT 42.9%); its gains are on logical-search tasks.

What would change this conclusion: a run where zeroing the feedback at
inference lowers accuracy. None of the five checkpoints tested does. What was
not tested: more than one epoch at the first stage, full fine-tuning, seeds
other than 42, and non-GSM8K tasks.

Recommendations:

1. Report the campaign as a negative result with these controls attached.
2. If the goal is a faster model on math, pursue the shortened-text direction
   with more training; it is as accurate as the latent version at a fraction
   of the complexity, and its accuracy gap is the thing to close.
3. If the goal is to validate the Coconut implementation, reproduce ProsQA or
   ProntoQA with a small model, where the paper reports clear gains.

## Artifact hashes (SHA-256 prefix)

| File (under `work/runs/`) | Prefix |
|---|---|
| `claude-stage1-diagnostics/s1-ablation-zero.jsonl` | `c1c3a6edfe52985a` |
| `claude-stage1-diagnostics/s1-ablation-shuffle.jsonl` | `d749d1e97971e582` |
| `claude-stage1/short-text-lr2e-5/validation/epoch-001.jsonl` | `de885865e2bb873a` |
| `claude-stage1/feedback-lr1e-4/validation/epoch-001.jsonl` | `f8a7803d658c18e8` |
| `claude-stage1-diagnostics/f1e4-ablation-zero.jsonl` | `451eb038010e477e` |
| `claude-stage1/feedback-aux0.5-lr2e-5/validation/epoch-001.jsonl` | `02669af3e3e1b317` |
| `claude-stage1-diagnostics/feedback-aux0.5-lr2e-5-ablation-zero.jsonl` | `42f15aa82c09ffa6` |

Teacher-forced diagnostics are the `tf-*.json` files in the diagnostics
directory. The campaign's own files keep the hashes recorded in
`coconut-gsm8k-20260905-v2/partial-results.json`.

# Latent steps improve accuracy over matched shortened-text training

**The hybrid candidate scored 96/100, compared with 66/100 for a separately trained
zero-latent model with the same shortened reasoning targets.** The candidate solved
30 additional questions and lost none. The paired accuracy difference is 30
percentage points, with a bootstrap 95% interval of [21, 39] points.

| Configuration | Correct | Arithmetic | Links |
| --- | ---: | ---: | ---: |
| Trained zero-latent control, shortened text | 66/100 | 19/53 | 47/47 |
| Two latent steps, then shortened text | 96/100 | 49/53 | 47/47 |
| Trained control, full text CoT mode | 99/100 | 52/53 | 47/47 |

The shortened-text control retains 99/100 when allowed full text reasoning. Its
compression failure is concentrated in arithmetic intermediate values. Both
shortened-text paths terminate without truncation; the control's 34 wrong results
are numeric answers, not missing answer delimiters.

## What is matched

Both runs start from the same corrected warmup checkpoint and use the same base
model, train/validation files, seed, maximum 400 updates, batch size 4, learning
rate 2e-5, alternating CoT/hybrid schedule, last-six-layer LoRA configuration,
one removed reasoning step, and fixed `Reasoning:` boundary. Configuration equality
was verified programmatically. The planned training difference is two latent
positions for the candidate versus zero for the control.

Each run selects its own checkpoint using hybrid validation loss. The candidate
selects step 300; the control selects step 400. Maximum updates and the sampling
recipe are matched, but selected-checkpoint exposure and training FLOPs are not
equal. Latent positions add computation during training. This is evidence under
the tested recipe, not proof that every zero-latent training strategy fails.

## Interpretation and limits

This comparison supports a practical contribution from latent computation on the
diagnostic pilot, beyond simply shortening supervised text. The implemented loop
passes continuous activations without intermediate vocabulary projection or token
sampling, then generates remaining text reasoning. The result concerns hybrid
latent/text reasoning and does not establish a new semantic language of thought.

These are the first 100 validation questions, including the 32 questions used
for checkpoint selection. One training seed and simple diagnostic tasks do not
establish broad reasoning quality. The sequential timings in these predictions
are not used to claim a speedup. See the separate
[interleaved timing report](../boundary-accuracy-latency/README.md) and
[results summary](../../docs/experiment-log.md) for subsequent evaluations.

## Artifacts

Control predictions, summaries, paired accuracy comparisons, verified training
configuration, selected checkpoint metadata, logs, and source snapshots are
included. The candidate predictions and shared validation questions are in the
[fixed-boundary report](../curriculum-boundary-stage1/README.md); their hashes are
recorded here. Exported metadata replaces absolute workspace prefixes and retains
original summary hashes. Commands are in the
[reproduction guide](../../docs/reproduce.md). Base and adapter weights are not
included in this report.

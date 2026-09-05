# Fixed-boundary pilot: 96/100 with hybrid latent and text reasoning

**Two continuous feedback steps followed by remaining text reasoning scored
96/100, versus 99/100 for retained text reasoning.** This meets the explicitly
accepted quality milestone for this pilot. It does not establish statistical
equivalence, a measured speedup, or general reasoning performance.

| Configuration | Correct | Mean generated tokens | Truncated |
| --- | ---: | ---: | ---: |
| Text CoT | 99/100 | 22.07 | 0 |
| 2 latent steps, then text | 96/100 | 15.64 | 0 |
| Zero feedback | 93/100 | 15.71 | 0 |
| Reversed feedback features | 90/100 | 16.51 | 1 |
| Repeated initial feedback | 86/100 | 19.07 | 4 |
| Hybrid decoder with 0 latent steps | 90/100 | 18.39 | 0 |

The normal hybrid path solved 49/53 arithmetic and 47/47 link problems. Text CoT
solved 52/53 and 47/47 respectively. The four hybrid failures involve incorrect
intermediate arithmetic or confusing reasoning stages. Scoring rules were unchanged.

## Paired accuracy evidence

Each difference below is normal hybrid accuracy minus the control on the same
100 questions. Intervals are paired bootstrap 95% intervals and are not adjusted
for examining multiple exploratory controls.

| Control | Difference in percentage points | Interval |
| --- | ---: | ---: |
| Text CoT | -3 | [-8, +1] |
| Zero feedback | +3 | [-3, +9] |
| Reversed feedback features | +6 | [-1, +13] |
| Repeated initial feedback | +10 | [+3, +17] |
| Hybrid decoder with 0 latent steps | +6 | [-1, +13] |

Zeroing or corrupting explicit feedback retains the attention caches. A zero
input can still attend to the prompt and write problem-dependent activations.
The zero-latent inference control instead removes positions from a checkpoint
trained with two positions. A separately trained zero-latent model with the same
shortened text targets remains necessary to assess the latent method's contribution.

## Mechanism and timing

The feedback loop passes continuous hidden activations through a learned bridge
into transformer input embeddings. It performs no vocabulary projection, token
sampling, or nearest-token lookup inside that loop. The model then generates
the remaining text reasoning and answer. This is hybrid latent/text reasoning,
not a fully language-free system or a reproduction of Astra's architecture.

The fixed boundary adds five forced text tokens after the two latent positions.
These tokens are included in measured time and position counts, although they
are excluded from generated-token counts. Fewer generated tokens alone do not
prove lower latency or computation. Raw sequential timing is retained in the
predictions for audit, but is not interpreted as a speedup. A separate repeated,
interleaved comparison against the 99/100 warmup CoT checkpoint is pending.

## Training and limitations

- Pinned Gemma 4 E2B 4-bit checkpoint, float32 computation, MLX 0.32.2 and MLX LM
  0.31.3, Apple M5 with 32 GB unified memory.
- Started from corrected warmup step 400. Trained 400 additional updates, batch
  size 4, learning rate 2e-5, alternating CoT and hybrid modes, seed 42. Removed
  the first annotated reasoning step from hybrid targets and inserted a fixed
  `Reasoning:` boundary after two latent positions. The boundary is masked out
  of the training loss. Prompt text contains no removed gold reasoning.
- Selected step 300 by hybrid loss on the first 32 validation questions. This
  development sample overlaps the 100-question evaluation; these results are
  exploratory validation evidence, not an independent test estimate.
- Last-six-layer LoRA plus bridge: 2,342,913 trainable parameters. Those layers
  reuse earlier attention keys/values, leaving cache-writing projections frozen.
- The 96/100 quality screen was accepted after observing this result. It replaces
  the earlier screen for advancing this pilot to timing, and is not a claim that
  the initial accuracy-matching objective was met.
- One training seed, short generated diagnostic tasks. Test and OOD remain
  unexamined. Public benchmark transfer and matched training controls are pending.

## Artifacts

Original predictions, exported summaries, paired accuracy comparisons, selected
questions, training logs/configuration, and verified source snapshots accompany
this report. Exported metadata replaces the absolute workspace prefix and retains
original summary hashes. The earlier warmup details are in
[the preceding pilot](../pilot-v2/README.md). Base and adapter weights are not
included in this report. No upstream proposal has been submitted.

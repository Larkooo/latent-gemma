# Staged pilot: the no-boundary transition did not retain accuracy

**Result: hybrid latent/text reasoning scored 62/100, compared with 99/100 for
text reasoning on the same checkpoint.** The paired accuracy difference was
-37 percentage points, with a bootstrap 95% interval of
[-47, -28] points.
The candidate fails the accuracy-matching target.

| Configuration | Correct | Median warm request seconds | Mean generated tokens | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Text CoT | 99/100 | 0.639 | 22.07 | 0 |
| 2 latent steps, then text | 62/100 | 0.437 | 18.34 | 0 |
| Zero feedback | 60/100 | 0.336 | 17.86 | 1 |
| Reversed feedback features | 59/100 | 0.574 | 20.68 | 4 |
| Repeated initial feedback | 68/100 | 0.496 | 20.06 | 2 |
| Hybrid decoder with 0 latent steps | 99/100 | 0.908 | 22.07 | 0 |

The normal latent path solved 28/53 arithmetic and 34/47 link problems. Text CoT
solved 52/53 and 47/47 respectively. Zeroing or repeating feedback did not expose
a reliable benefit from evolving latent states. No scoring rules were changed.

## Timing limitation

The CoT and zero-latent hybrid paths generated identical text on every question,
yet their sequential-run median timings differed substantially. This demonstrates
that these timings do not support a robust speedup claim. Future timing comparisons
interleave conditions in randomized, counterbalanced order, repeat each condition,
and count each question once for accuracy. Zero/repeat ablations also change graph
dependencies that lazy execution can prune; nominal positions are not equal FLOPs.

Warm request timing includes prompt formatting/tokenization, prefill, latent
computation, generation through stop or cap, final decoding, and answer extraction.
It excludes model loading, warmup, and external serving or network overhead.

## Training and scope

- Pinned Gemma 4 E2B 4-bit checkpoint, float32 computation, MLX 0.32.2 and MLX LM
  0.31.3, Apple M5 with 32 GB unified memory.
- Started from corrected warmup step 400. Trained another 400 updates, batch size
  4, learning rate 2e-5, alternating CoT and hybrid modes. Two continuous positions
  replace the first annotated text-reasoning step. No fixed text boundary follows
  the continuous positions. Selected checkpoint: step 400 by hybrid validation loss.
- Last-six-layer LoRA plus bridge: 2,342,913 trainable parameters. All six layers
  reuse earlier attention keys/values, leaving cache-producing projections frozen.
- Sampler replay shows 800 hybrid draws covering 761 unique training examples.
  These are short pilot budgets. The failure does not establish that longer or
  differently parameterized training cannot work.
- First 100 validation examples only, one training seed. Test/OOD, public-benchmark,
  and matched extra-training evaluations were not performed for this recipe.
- A ten-example intermediate audit ran alongside training. Its inference timings
  are excluded, and training wall time should not be treated as an isolated cost
  benchmark. The final validation matrix ran serially after training.

## Follow-up and artifacts

The [fixed-transition experiment](../curriculum-boundary-stage1/README.md) added
a `Reasoning:` prefix between latent states and generated text while preserving
the warmup checkpoint and other training settings.

Original predictions, exported summaries, paired comparisons, selected questions,
training logs/configuration, and verified source snapshots accompany this report.
Exported metadata replaces the absolute workspace prefix and retains original
summary hashes. The frozen warmup source and earlier training details are in
[the preceding pilot](../pilot-v2/README.md). Base and adapter weights are not
included.

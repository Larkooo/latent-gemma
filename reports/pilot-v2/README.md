# Diagnostic pilot: direct compression did not retain accuracy

**Result: the first four-step feedback recipe failed the quality target.** On the
same 100 validation questions, text reasoning scored 99/100 and latent reasoning
scored 80/100. The latent path was faster, but lost 19 percentage points of
accuracy. Its paired bootstrap 95% interval for the accuracy difference was
[-27, -12] percentage points. This does not satisfy the prespecified two-point
accuracy-retention target.

Zeroing feedback scored 81/100 and repeating the initial feedback scored 80/100.
The three-point gain over the same checkpoint's direct-answer mode had a paired
95% interval of [-3, 9] points. These results do not establish a useful contribution
from evolving feedback states. They motivate a different training recipe.

## Measured results

All rows below use the same pinned Gemma 4 E2B 4-bit checkpoint, float32
computation, greedy decoding, and the first 100 diagnostic validation examples
(53 arithmetic, 47 link traversal). None of these fine-tuned runs truncated.

| Configuration | Correct | Median seconds | Mean generated tokens |
| --- | ---: | ---: | ---: |
| Warmup, direct answer | 72/100 | 0.112 | 2.90 |
| Warmup, text reasoning | 99/100 | 0.761 | 22.07 |
| Mixed checkpoint, direct answer | 77/100 | 0.114 | 2.82 |
| Mixed checkpoint, text reasoning | 99/100 | 0.717 | 22.07 |
| Mixed checkpoint, 4 latent steps | 80/100 | 0.244 | 2.83 |
| 4 zero-input latent positions | 81/100 | 0.133 | 2.83 |
| 4 steps, reversed feedback features | 78/100 | 0.240 | 2.83 |
| 4 steps, repeated initial feedback | 80/100 | 0.184 | 2.83 |

The arithmetic gap dominates: four latent steps solved 33/53 arithmetic problems,
compared with 52/53 for text reasoning. Both solved all 47 link problems. The
latent arithmetic result was 13/29 for sum-then-product and 20/24 for
product-then-sum. Thus these tasks do not yet support a broad reasoning claim.

## Setup and limitations

- Apple M5 with 32 GB unified memory; MLX 0.32.2, MLX LM 0.31.3.
- Base: `mlx-community/gemma-4-e2b-it-4bit`, revision
  `238767527555cb75a05732a84dff5d6ba0dd6809`.
- Frozen quantized base, LoRA on the last six layers, rank 16, plus a trainable
  continuous feedback bridge. There are 2,342,913 trainable parameters.
- Warmup: 400 updates alternating direct/CoT modes. Mixed run: another 600 updates
  alternating direct/CoT/latent modes. Batch size 4, learning rate 2e-5. Selected
  checkpoints are warmup step 400 and mixed step 500, using validation token loss.
- The retained text-reasoning mode still scored 99/100 on the mixed checkpoint.
  The problem is compression accuracy, not loss of this text-mode capability.
- Ablations match nominal latent positions, not measured FLOPs. Constant or
  repeated inputs remove value dependencies that MLX's lazy evaluation may prune.
  Their shorter latency is not proof of equal computation at higher speed.
- All observations are exploratory validation results from one training seed.
  The final test split and longer/larger-input OOD split remain unexamined.
- A matched additional-training control and public-benchmark evaluation remain
  necessary for any later positive claim. The current accuracy gap is already
  large enough to reject this recipe against the quality target.
- The pretrained forced-format pilots are included in the raw files for
  transparency: direct scored 22/100 with 69 truncations at 16 tokens; explicit
  CoT scored 14/100 with 83 truncations at 96 tokens. These are format/budget
  failures and must not be presented as released-model capability estimates.
  Native thinking and ordinary chat are evaluated separately at larger budgets.

## Artifacts and next experiment

`results.csv` contains exact summary values. The JSONL files contain the original
predictions and timing measurements. Their SHA-256 values match the original
summary records. Exported summaries replace the absolute workspace prefix and
record the original summary hash. Source snapshots, training configurations,
loss logs, dataset manifest, and the selected 100 examples accompany the results.
No base weights or adapter weights are included in this report.

The next experiment gradually replaces initial text reasoning steps with latent
positions, retaining the remaining text as a training target. This is motivated
by [Coconut's training curriculum](https://arxiv.org/html/2412.06769v2), but uses
Gemma, LoRA, and a different feedback bridge. It is a new hypothesis to test;
this pilot is not evidence that it will succeed. No upstream proposal has been
submitted.

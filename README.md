# Latent Gemma

Continuous-state reasoning experiments with Gemma and MLX. A learned bridge feeds
hidden states back into the transformer, replacing part of a text reasoning
trace. The model then generates the remaining reasoning and answer.

Author and maintainer: [Larkooo](https://github.com/Larkooo).

Gemma discussion: [Gemma 4 hybrid continuous-state reasoning](https://github.com/google-deepmind/gemma/discussions/795).

## Results

The current Gemma 4 E2B experiment uses two latent positions followed by a fixed
text transition. On 600 held-out diagnostic arithmetic and link-traversal questions:

| Method | Accuracy | Median answer latency |
|---|---:|---:|
| Text reasoning | 598/600 (99.67%) | 1.031 s |
| Hybrid latent/text reasoning | 583/600 (97.17%) | 0.922 s |

The hybrid reduced median completed-answer latency by **10.6%**, with a
**2.5 percentage-point accuracy loss**. Two repeats per method and question
produced 2400 timed requests, with randomized, counterbalanced order on an Apple
M5 with 32 GB memory. The speed ratio was 1.118, with a paired-question bootstrap
95% interval of [1.069, 1.152]. The recipe was fixed before this test.
Mean completed-answer latency was 1.195 s versus 0.994 s, a 16.8% reduction.

The earlier 100-question validation pilot scored 96/100 versus 99/100 with 17.6%
lower latency. A separately trained control using the same shortened text targets
without latent positions scored 66/100 on that validation sample. Results use one
training seed and two procedural task families. On the completed harder OOD
arithmetic split, accuracy fell from 190/200 (95%) to 161/200 (80.5%); link
accuracy was unchanged at 199/200. These results do not isolate a benefit from
feedback content or establish broader reasoning gains or lower total computation.

A new [three-seed GSM8K curriculum campaign](docs/coconut-campaign.md) compares
recurrent feedback with trained pause inputs, shortened text, and full text
reasoning. Its training recipe and evaluation policy are separate from the
diagnostic results above.

[Independent test report](reports/independent-test/README.md) ·
[OOD report](reports/ood-test/README.md) ·
[Experiment results](docs/experiment-log.md) ·
[All reports](reports/README.md)

## Setup

Python 3.12+ and Apple Silicon are required.

```sh
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The [reproduction guide](docs/reproduce.md) covers the pinned model, training,
combined accuracy/latency benchmark, controls, and public benchmark data.

## Implementation

```text
prompt → K continuous feedback steps → remaining reasoning → answer
```

Latent steps use continuous embeddings directly, without token sampling or a
vocabulary projection. LoRA adapts the backbone. Gemma 4 latent positions retain
the continuous per-layer embedding branch and use zero token-table contribution.

Current performance results use Gemma 4 E2B. The backend also supports Gemma 3
text experiments. See the [method and evaluation protocol](docs/protocol.md)
for cache handling, training, measurement scope, and limitations.

## Repository

- `src/latent_gemma/` — model wrapper, training, inference, and evaluation.
- `scripts/` — model/data preparation, paired benchmarks, and result utilities.
- `tests/` — model semantics, gradients, decoding, scoring, and data checks.
- `reports/` — frozen predictions, timing traces, source snapshots, and analyses.

For Gemma integration, see the [continuous-input proposal](docs/upstream-proposal.md).
The approach builds on [Coconut](https://arxiv.org/abs/2412.06769).

Code is available under the [MIT license](LICENSE). Model weights are obtained
separately and retain their upstream terms.

# Latent Gemma

Continuous-state reasoning experiments with Gemma and MLX. A learned bridge feeds
hidden states back into the transformer, replacing part of a text reasoning
trace. The model then generates the remaining reasoning and answer.

## Results

The current Gemma 4 E2B pilot uses two latent positions followed by a fixed text
transition. On 100 diagnostic arithmetic and link-traversal questions:

| Method | Accuracy | Median answer latency |
|---|---:|---:|
| Text reasoning | 99/100 | 1.339 s |
| Hybrid latent/text reasoning | 96/100 | 1.103 s |

The hybrid reduced median completed-answer latency by **17.6%**. Measurements
used three repeats per method and question, alternating methods on an Apple M5
with 32 GB memory. The speed ratio was 1.214, with a paired-question bootstrap
95% interval of [1.105, 1.376].

A separately trained control using the same shortened text targets without latent
positions scored 66/100. These are exploratory validation results from one seed.
Independent test and OOD evaluation is running. Broader reasoning gains and lower
total computation have not been established.

[Accuracy and latency report](reports/boundary-accuracy-latency/README.md) ·
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

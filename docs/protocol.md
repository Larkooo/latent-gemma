# Method and evaluation

## Continuous feedback

The model reads a prompt, maps its final hidden state through a learned bridge,
and feeds the resulting continuous embedding into a new transformer position.
This repeats for a fixed number of latent steps. A fixed text transition then
starts the remaining reasoning and answer.

The bridge applies RMS normalization, a low-rank residual projection, and a
learned gain initialized from the backbone's embedding scale. Latent positions
occupy real attention-cache positions. The loop uses no vocabulary projection,
token sampling, or nearest-token lookup.

Gemma 4 normally uses token IDs to construct part of its per-layer inputs. At
latent positions, this implementation zeros that token-table contribution and
preserves the continuous projection branch and native mixing scale. Ordinary
tokens use the original embedding path.

The implementation reuses the backbone's transformer and attention masks with
differentiable, nonrotating caches. Gemma 4 shared KV layers reuse the appropriate
earlier cache. Nominal input positions do not imply that every layer physically
executes: MLX can eliminate work whose outputs are unused.

## Training

The backbone is frozen except for LoRA adapters. The current pilot adapts the
last six layers at rank 16 and uses a rank-64 feedback bridge. LoRA targets query,
value (where present), output, and MLP down projections.

A direct-answer/text-CoT warmup is followed by alternating full-CoT and hybrid
batches. Hybrid targets omit the first annotated reasoning step. The removed
step is never supplied in the inference prompt. Boundary tokens are encoded
separately, matched exactly between training and inference, and masked from loss.

Gemma 4 uses float32 computation for stable recurrent gradients while retaining
quantized weight storage. Tests cover causal target alignment, cached versus
full-sequence gradients, shared-KV behavior, checkpoint loading, and text-path
equivalence.

## Data and selection

Diagnostic data contains arithmetic expressions and directed-link traversal.
Train/validation/test splits use semantic IDs; OOD examples use larger operands
or longer paths. GSM8K preparation preserves the original test split and draws
validation only from its training data.

Checkpoint selection uses validation loss. The reported 100-question pilot is
exploratory validation and overlaps checkpoint selection. Its 96/100 accuracy was
accepted as a development target after observing the result; it is not a
prespecified equivalence margin. Independent test and OOD runs freeze the model,
data, source, and decoding settings before evaluation.

## Measurement

The first evaluation after training compares accuracy and latency together using
`scripts/benchmark_pair.py`. Both methods use the same base checkpoint, numerical
dtype, decoder, questions, and token cap. Conditions are interleaved with
randomized, counterbalanced order. Each request starts with fresh caches.

`end_to_end_latency_s` measures a warm request from prompt formatting through
tokenization, prefill, latent computation, forced prefixes, generation, final
decoding, and answer extraction. GPU work is synchronized. Model loading, warmup,
external serving, and network time are excluded. The narrower `latency_s` field
excludes prompt preparation and final decoding; missing historical measurements
are never synthesized.

Repeated outputs must agree. Each question contributes once to accuracy and uses
its median time across repeats. Paired bootstrap intervals describe variation
across questions, not drift between hardware sessions. Report all questions,
token budgets, truncations, median/p95 latency, and work counters.

Numeric answers use exact finite decimal equality. Link labels use exact string
matching. Scoring corrections retain original records and apply the same named
policy to every compared condition.

## Controls and interpretation

Full text reasoning provides the quality reference. A separately trained
zero-latent control uses the same shortened targets and maximum update budget.
Inference ablations zero, reverse, or repeat feedback features while retaining
the attention cache. Since the cache can still carry problem-dependent states,
these ablations do not remove every source of latent computation.

Shorter text, lower latency, fewer FLOPs, and lower energy are different claims.
Latent positions and forced transitions consume work, and batched positions have
different costs from autoregressive steps. The current counters are logical
accounting, not hardware FLOP or energy measurements.

The prototype retains some text reasoning. Visible explanations and activation
ablations do not establish faithful introspection or safety. Small diagnostic
results do not establish general reasoning superiority.

## Related work

- [Coconut](https://arxiv.org/abs/2412.06769): continuous thoughts and staged
  replacement of text reasoning.
- [Recurrent-depth language models](https://arxiv.org/abs/2502.05171): recurrence
  through a shared internal block, which differs from this embedding-feedback loop.
- [Gemma](https://github.com/google-deepmind/gemma) and
  [MLX LM](https://github.com/ml-explore/mlx-lm): backbone implementations.

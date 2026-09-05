# Proposal: continuous embedding inputs for Gemma

## Motivation

Continuous-state reasoning experiments need to read hidden states and feed them
back as embeddings with correct positions, masks, and caches. A token-only public
interface requires researchers to copy model internals. Gemma 4 adds a second
concern: part of its per-layer input normally depends on discrete token IDs.

The proposed contribution is a supported continuous-input path and a small
research example using it.

## Prototype evidence

An MLX/LoRA prototype on Gemma 4 E2B scored 96/100 versus 99/100 for its text-CoT
reference, with median completed-answer latency of 1.103 versus 1.339 seconds.
The comparison used 600 interleaved requests. The speed ratio was 1.214 with a
paired-question 95% interval of [1.105, 1.376]. A separately trained shortened-text
control without latent positions scored 66/100.

These are single-seed diagnostic validation results. The prototype retains text
reasoning, its public-benchmark transfer pilot does not establish a gain, and
the completed OOD test shows a 14.5-point arithmetic accuracy loss relative to
text reasoning. Independent diagnostic test accuracy was 583/600 versus 598/600.
Lower total inference FLOPs have not been
measured. See the [results](experiment-log.md) and
[raw timing report](../reports/boundary-accuracy-latency/README.md).

## Interface

At inspected commit `7b785991bd78626c73b317eb43fdbb6c292f7b9c`, Gemma 4's
token-based `Transformer.__call__` already supports hidden-state returns. Its
private `_Inputs` structure and `_apply_attention` method provide the machinery
to reuse for an embedding-input path.

The public contract should specify:

- Whether embeddings are scaled, and where native input scaling occurs.
- How per-layer inputs are constructed when token IDs are absent.
- Position, global/sliding-mask, and cache-update behavior.
- Hidden-state output without requiring vocabulary logits.

The MLX prototype uses unscaled continuous embeddings, preserves the projected
per-layer branch, and zeros the token-table contribution at latent positions.
It retains the native mixing scale. This is an explicit experimental policy;
released weights were not trained specifically for this input path.

## Implementation and review

The official implementation would follow the repository's JAX/Flax conventions
and reuse its transformer internals. Normal token and multimodal behavior must
remain compatible. Validation should cover token/embedding equivalence, shared
KV, sliding masks, multiple cached positions, and gradients through feedback.
The current MLX tests do not validate a JAX/Flax port.

The first discussion should establish whether a community example, experimental
utility, or general input interface fits the library. An agreed interface can
then be implemented and submitted as a focused PR. A library contribution does
not modify official Gemma weights.

[Gemma contribution requirements](https://github.com/google-deepmind/gemma/blob/main/CONTRIBUTING.md)
require review and a Google CLA. The underlying continuous-thought approach is
established prior work; see [Coconut](https://arxiv.org/abs/2412.06769).

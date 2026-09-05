# Proposed contribution: continuous embedding inputs for Gemma research

Status: local working draft. No issue or pull request has been submitted. Measured
results and final scope must be added before this is proposed upstream.

## Problem

Researchers exploring continuous-state reasoning need to read hidden states and
feed continuous embeddings back through the transformer with correct positions,
attention masks, and caches. A token-only forward API forces them to copy model
internals. Gemma 4 E2B/E4B add a second concern: per-layer embeddings normally
depend on discrete token IDs, which continuous latent positions do not have.

## Intended behavior

An explicitly experimental input path accepts embeddings and the corresponding
position/mask metadata. Normal token inputs preserve existing behavior. The
per-layer embedding policy is explicit rather than silently recovering the
nearest vocabulary token. Hidden-state outputs support a separately implemented
experimental sampler and training recipe.

Our MLX prototype retains the projected, continuous per-layer input branch and
sets the token-indexed contribution to zero at latent positions. It keeps the
native mixing scale. That is a research design decision to evaluate, not a
claim that released weights were trained for such inputs.

## Evidence required before submission

- Published, pinned base checkpoint and adapter configuration.
- Train/validation/test hashes and independent evaluation of final settings.
- Accuracy and measured latency compared with explicit CoT and native Gemma
  thinking, plus matched direct-answer and training-budget controls.
- Evidence from activation ablations; positive and negative results.
- Tests for normal-path equivalence, cache/position correctness, causal masking,
  gradients across latent steps, and per-layer embedding handling.
- Clear scope: experiments on text inputs; no claim about multimodal reasoning
  unless separately evaluated. No claim to reproduce another lab's architecture.

## Contribution sequence

1. Share the standalone experiment and evidence in a maintainer discussion.
2. Agree whether an example, experimental utility, or general embedding-input
   interface belongs in the library.
3. Port the agreed interface to the repository's JAX/Flax conventions, keeping
   the main sampling path unchanged and adding focused tests.
4. Submit a regular PR with the measured behavior, motivation, and validation.

The repository welcomes contributions and requires review and a Google CLA.
An accepted library change does not change the official released model weights.

References:
- https://github.com/google-deepmind/gemma/blob/main/CONTRIBUTING.md
- https://github.com/google-deepmind/gemma/blob/7b785991bd78626c73b317eb43fdbb6c292f7b9c/gemma/gm/nn/gemma4/_transformer.py
- https://developers.googleblog.com/en/unlock-global-communication-gemma-projects/


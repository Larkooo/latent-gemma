# Proposed contribution: continuous embedding inputs for Gemma research

Status: local working draft. No issue or pull request has been submitted. The
first combined validation accuracy/latency result is included. Independent test
results and final scope remain outstanding.

## Current evidence

The standalone MLX implementation passes continuous activations through a learned
bridge into new transformer positions, then generates remaining text reasoning.
On 100 diagnostic validation questions, the fixed-boundary hybrid scored 96/100
versus 99/100 for text CoT. A separately trained control with the same shortened
text targets and maximum update budget scored 66/100, while retaining 99/100 in
full text mode. The hybrid solved 30 additional questions and lost none, with a
paired bootstrap accuracy-difference interval of [21, 39] percentage points.
See the [matched-training report](../reports/matched-short-text-control/README.md).

The combined accuracy/latency comparison retained 96/100 versus 99/100 while
measuring median warm completed-answer latency of 1.103 seconds versus 1.339
seconds: 17.6% lower median latency, or a 1.214 speed ratio. Both methods answered
the same 100 questions three times, with randomized, counterbalanced order, fresh
caches, and the same serial decoder. The speed-ratio 95% paired-question interval
was [1.105, 1.376]. The benefit was concentrated in arithmetic; link-question
latencies were approximately equal. See the
[combined report](../reports/boundary-accuracy-latency/README.md).

This supports useful latent computation and a measured pilot latency benefit
under the tested recipe. It is one seed, one Apple Silicon session, simple
diagnostic tasks, and exploratory validation that overlaps checkpoint selection.
Training FLOPs are not matched, and lower inference FLOPs have not been measured.
Initial GSM8K transfer scored 24/40 for the hybrid, 24/40 for warmup text CoT, and
28/40 for the candidate adapter in full-text mode; no adapter was trained on
GSM8K. Independent diagnostic test/OOD evaluation is in progress. An upstream
submission must not generalize the pilot speedup into a broad model-quality claim.

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

At inspected Gemma commit `7b785991bd78626c73b317eb43fdbb6c292f7b9c`, Gemma 4
already exposes `return_hidden_states` on its token-based `Transformer.__call__`.
The internal `_Inputs` structure carries embeddings, positions, global/sliding
masks, input masks, and per-layer inputs into `_apply_attention`. The concrete
extension to discuss is a supported continuous-input path reusing that machinery,
rather than another transformer implementation. Whether it is a separate method
or a general input option should follow the maintainers' API preference.

The contract must specify embedding scaling, per-layer input construction when
token IDs are absent, cache updates, and hidden-state versus vocabulary outputs.
The existing method constructs vocabulary logits even when hidden states are
requested; actual execution can depend on compiler elimination of unused results.
A research-facing hidden-only path should make that intent explicit without
relying on a caller's compilation context.

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

The MLX tests establish prototype behavior; they do not validate a JAX/Flax port.
Any proposed Gemma implementation needs its own token-versus-embedding equivalence
tests, shared-cache and sliding-mask checks, and differentiation through multiple
continuous positions. Existing token and multimodal entry points should preserve
their current behavior.

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

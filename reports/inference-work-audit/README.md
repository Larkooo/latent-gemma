# Inference work in the boundary pilot

The hybrid emits 29.1% fewer tokens, but its mean nominal transformer positions
increase by 0.9%. Fewer emitted tokens alone therefore do not establish lower
total computation. This is a post-observation accounting audit of the frozen
[100-question diagnostic pilot](../curriculum-boundary-stage1/README.md), with the
same selected adapter in full-text and two-step hybrid modes.

| Questions | Full-text generated tokens | Hybrid generated tokens | Full-text nominal positions | Hybrid nominal positions |
|---|---:|---:|---:|---:|
| All 100 | 22.07 | 15.64 | 62.85 | 63.42 |
| Arithmetic, 53 | 29.89 | 19.53 | 62.04 | 58.68 |
| Links, 47 | 13.26 | 11.26 | 63.77 | 68.77 |

Counts are means per question. Generated tokens include EOS. All 200 outputs
completed within the cap. The hybrid adds two continuous positions and a fixed
five-token transition. That transition is processed as one block; it does not
require five sequential sampling decisions. Arithmetic removes enough text to
reduce nominal positions; the short link answers do not.

The archived serial decoder makes one vocabulary projection per generated token
and none within the latent loop, so it also performs 29.1% fewer vocabulary
projections. This is code-derived accounting, not a hardware measurement. The
cost of transformer positions, batched transition tokens, attention, and the
vocabulary head differs. Lazy execution and shared KV also affect which
operations actually execute. Neither token counts nor nominal positions are
total FLOPs, energy, or a latency measurement.

The result identifies a concrete constraint for further compression: the
transition and latent overhead must be amortized over enough omitted text.
Repeated, interleaved timing remains the speed test. This report makes no speedup
or lower-total-compute claim.

Reproduce the accounting from the repository root:

```sh
python reports/inference-work-audit/audit.py
```

The script checks paired IDs, targets, prompt lengths, completed outputs, and the
recorded position equation. `audit.json` pins both input prediction files by hash.

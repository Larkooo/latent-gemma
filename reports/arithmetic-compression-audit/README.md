# Arithmetic continuation audit

**The latent candidate produced the correct final answer and exactly the intended
shortened equation on 46/53 arithmetic questions, versus 16/53 for the trained
zero-latent control.** The 30-question advantage therefore also appears under this
stricter output-compression metric.

| Model | Correct answer | Correct answer with exact shortened equation |
| --- | ---: | ---: |
| Two latent steps, then text | 49/53 | 46/53 |
| Trained zero-latent control | 19/53 | 16/53 |

Three correct continuations from each model did not match the shortened target.
The latent candidate regenerated the full two-equation solution twice and added
an extra equation once. The control regenerated the full solution three times.
The 96/100 overall answer score therefore does not mean every response strictly
omitted the first text reasoning step. Actual generated text and its cost remain
included in the latency comparisons.

The metric requires a correct final answer and a continuation equal to the second
annotated equation, ignoring whitespace and trailing periods. It is an audit of
visible output compression, not a probe of what any latent activation represents.
Every arithmetic question is included in the denominator; questions are not
selected based on which model answered correctly.

`audit.py` reproduces `audit.json` from the immutable predictions and validation
sample in the adjacent reports. It records their hashes and every inspected
continuation. The [matched-training report](../matched-short-text-control/README.md)
contains the training comparison and its limitations.

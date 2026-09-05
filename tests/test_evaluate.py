import mlx.core as mx

from latent_gemma.data import Example
from latent_gemma.evaluate import generate


def test_hybrid_runs_latent_steps_then_decodes_without_forcing_answer_prefix():
    class Model:
        def eval(self):
            pass

        def prefill(self, prompt, steps, ablation):
            assert steps == 3
            assert ablation == "zero"
            return mx.zeros((1, 1, 2)), []

        def logits(self, state):
            return mx.array([[[0.0, 1.0]]])

        def hidden(self, *args, **kwargs):
            raise AssertionError("Hybrid decoding must not force an answer prefix")

    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "prompt"

        def encode(self, text, **kwargs):
            assert text == "promptReasoning: "
            return [0]

        def decode(self, ids):
            return ""

    example = Example("test", "links", "Question", "A -> B.", "B", "validation")
    row = generate(Model(), Tokenizer(), example, "hybrid", 3, ablation="zero")
    assert row["latent_steps"] == 3
    assert row["forced_tokens"] == 0
    assert row["terminated"]

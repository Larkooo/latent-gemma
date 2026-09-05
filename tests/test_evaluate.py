import mlx.core as mx
import pytest

from latent_gemma.data import Example
from latent_gemma.evaluate import generate


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("mode,latent_steps", [("hybrid", 3), ("plain", 0)])
def test_unforced_decoding_respects_latent_count_and_prompt(mode, latent_steps):
    class Model:
        def eval(self):
            pass

        def prefill(self, prompt, steps, ablation):
            assert steps == latent_steps
            assert ablation == "zero"
            return mx.zeros((1, 1, 2)), []

        def logits(self, state):
            return mx.array([[[0.0, 1.0]]])

        def hidden(self, *args, **kwargs):
            raise AssertionError("This decoding mode must not force an answer prefix")

    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "prompt"

        def encode(self, text, **kwargs):
            assert text == ("prompt" if mode == "plain" else "promptReasoning: ")
            return [0]

        def decode(self, ids):
            return ""

    example = Example("test", "links", "Question", "A -> B.", "B", "validation")
    row = generate(Model(), Tokenizer(), example, mode, 3, ablation="zero")
    assert row["latent_steps"] == latent_steps
    assert row["forced_tokens"] == 0
    assert row["terminated"]

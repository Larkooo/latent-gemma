import mlx.core as mx
import pytest

from latent_gemma.data import Example, encode_example
from latent_gemma.evaluate import generate, summarize


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


def test_hybrid_boundary_matches_training_and_measures_complete_request(monkeypatch):
    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def encode(self, text, **kwargs):
            return {
                "promptReasoning: ": [0],
                "\nReasoning: ": [2, 3],
                "5 * 4 = 20.\nAnswer: 20": [4, 5],
                # Encoding across the boundary would merge the first content
                # token, unlike the forced prefix used during generation.
                "\nReasoning: 5 * 4 = 20.\nAnswer: 20": [2, 6, 5],
            }[text]

        def decode(self, ids):
            return ""

    class Model:
        def eval(self):
            pass

        def prefill(self, prompt, steps, ablation):
            assert prompt.tolist() == [[0]]
            assert steps == 2
            return mx.zeros((1, 1, 2)), []

        def hidden(self, ids, cache):
            assert ids.tolist() == [[2, 3]]
            return mx.zeros((1, 2, 2))

        def logits(self, state):
            return mx.array([[[0.0, 1.0]]])

    example = Example("test", "arithmetic", "Question", "2 + 3 = 5. 5 * 4 = 20.", "20", "train")
    prompt, target, mask = encode_example(Tokenizer(), example, "hybrid", 1, "reasoning")
    assert prompt == [0]
    assert target == [2, 3, 4, 5, 1]
    assert mask == [0.0, 0.0, 1.0, 1.0, 1.0]
    ticks = iter([10.0, 11.0, 14.0, 15.0])
    monkeypatch.setattr("latent_gemma.evaluate.time.perf_counter", lambda: next(ticks))
    row = generate(Model(), Tokenizer(), example, "hybrid", 2, hybrid_boundary="reasoning")
    assert row["forced_tokens"] == 2
    assert row["latency_s"] == 3.0
    assert row["end_to_end_latency_s"] == 5.0
    assert row["hybrid_boundary"] == "reasoning"


def test_summary_does_not_invent_missing_end_to_end_measurements():
    row = {
        "task": "links",
        "correct": True,
        "latency_s": 1.0,
        "generated_tokens": 2,
        "transformer_positions": 5,
        "terminated": True,
    }
    measured = {**row, "end_to_end_latency_s": 2.0}
    assert summarize([measured])["overall"]["median_end_to_end_latency_s"] == 2.0
    assert "median_end_to_end_latency_s" not in summarize([row])["overall"]
    assert "median_end_to_end_latency_s" not in summarize([row, measured])["overall"]

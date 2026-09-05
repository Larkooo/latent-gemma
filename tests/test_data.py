import pytest

from latent_gemma.data import (
    Example,
    encode_example,
    extract_answer,
    generate_dataset,
    read_examples,
    remaining_reasoning,
)
from latent_gemma.train import make_batch


def test_splits_are_reproducible_and_disjoint(tmp_path):
    first = generate_dataset(tmp_path / "a", 100, 40, 40, 20, seed=123)
    second = generate_dataset(tmp_path / "b", 100, 40, 40, 20, seed=123)
    assert first == second
    all_ids = []
    for split in ("train", "validation", "test", "ood"):
        examples = read_examples(tmp_path / "a" / f"{split}.jsonl")
        assert all(x.split == split for x in examples)
        all_ids.extend(x.id for x in examples)
    assert len(all_ids) == len(set(all_ids))


@pytest.mark.parametrize(
    "text,task,mode,expected",
    [
        ("20", "arithmetic", "direct", "20"),
        ("We compute 2 + 3.\nAnswer: 20.", "arithmetic", "cot", "20"),
        ("We compute 20.", "arithmetic", "cot", None),
        ("I think B", "links", "latent", None),
        ("B.", "links", "latent", "B"),
        ("Answer: 1,234", "gsm8k", "cot", "1234"),
        ('<|channel>thought\nUse "Answer:".<channel|>Answer: N', "links", "native", "N"),
        ("<|channel>thought\nAnswer: 12", "arithmetic", "native", None),
        ("**Answer: ** 12", "arithmetic", "native", "12"),
    ],
)
def test_answer_extraction(text, task, mode, expected):
    assert extract_answer(text, task, mode) == expected


def test_ragged_prompt_batch_rejected():
    with pytest.raises(ValueError, match="equal unpadded"):
        make_batch([([1, 2], [3], [1.0]), ([1], [3], [1.0])], 0)


@pytest.mark.parametrize("mode", ["direct", "latent"])
def test_first_answer_token_is_supervised_despite_whitespace_merging(mode):
    class MergingTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def encode(self, text, **kwargs):
            return {
                "promptReasoning: ": [1, 2],
                "\nAnswer: ": [3, 4, 5],
                "\nAnswer: B": [3, 4, 6],
                "B": [7],
            }[text]

    example = Example("example", "links", "Follow links", "A -> B.", "B", "train")
    _, target, mask = encode_example(MergingTokenizer(), example, mode)
    assert target == [3, 4, 5, 7, 99]
    assert mask == [0.0, 0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "task,reasoning,drop,expected",
    [
        ("arithmetic", "2 + 3 = 5. 5 * 4 = 20.", 1, "5 * 4 = 20."),
        ("arithmetic", "2 + 3 = 5. 5 * 4 = 20.", 2, ""),
        ("links", "A -> B -> C -> D.", 1, "B -> C -> D."),
        ("links", "A -> B -> C -> D.", 2, "C -> D."),
        ("links", "A -> B -> C -> D.", 3, ""),
        (
            "gsm8k",
            "First calculation.\n\nSecond calculation.\nFinal calculation.",
            1,
            "Second calculation.\nFinal calculation.",
        ),
    ],
)
def test_curriculum_removes_only_initial_reasoning_steps(task, reasoning, drop, expected):
    example = Example("test", task, "Question", reasoning, "answer", "train")
    assert remaining_reasoning(example, drop) == expected
    assert remaining_reasoning(example, 0) == reasoning
    assert remaining_reasoning(example, 100) == ""


def test_hybrid_supervises_remaining_text_without_leaking_removed_steps():
    class CharacterTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(map(ord, text))

    example = Example("test", "arithmetic", "Question", "2 + 3 = 5. 5 * 4 = 20.", "20", "train")
    prompt, target, mask = encode_example(CharacterTokenizer(), example, "hybrid", 1)
    assert "2 + 3 = 5" not in "".join(map(chr, prompt))
    assert "".join(map(chr, target[:-1])) == "5 * 4 = 20.\nAnswer: 20"
    assert mask == [1.0] * len(target)
    _, final, final_mask = encode_example(CharacterTokenizer(), example, "hybrid", 2)
    assert "".join(map(chr, final[:-1])) == "\nAnswer: 20"
    assert final_mask == [1.0] * len(final)

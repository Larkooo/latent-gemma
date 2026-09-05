import pytest

from latent_gemma.data import (
    Example,
    encode_example,
    extract_answer,
    generate_dataset,
    read_examples,
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

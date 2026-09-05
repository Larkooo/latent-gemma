"""Deterministic diagnostic tasks with semantic train/test separation."""

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    task: str
    question: str
    reasoning: str
    answer: str
    split: str


def semantic_id(task: str, key: str) -> str:
    return hashlib.sha256(f"{task}:{key}".encode()).hexdigest()[:20]


def assigned_split(identifier: str) -> str:
    value = int(identifier[:8], 16) % 10
    return "test" if value == 0 else "validation" if value == 1 else "train"


def make_candidate(rng: random.Random, task: str, ood: bool = False) -> Example:
    if task == "arithmetic":
        low, high = (20, 50) if ood else (1, 20)
        a, b, c = [rng.randint(low, high) for _ in range(3)]
        kind = rng.randrange(2)
        if kind == 0:
            question = f"Compute ({a} + {b}) * {c}."
            reasoning = f"{a} + {b} = {a + b}. {a + b} * {c} = {(a + b) * c}."
            answer = str((a + b) * c)
        else:
            question = f"Compute {a} * {b} + {c}."
            reasoning = f"{a} * {b} = {a * b}. {a * b} + {c} = {a * b + c}."
            answer = str(a * b + c)
        identifier = semantic_id(task, f"{kind}:{min(a, b)}:{max(a, b)}:{c}")
    elif task == "links":
        length = rng.randint(5, 6) if ood else rng.randint(2, 4)
        nodes = rng.sample(list("ABCDEFGHIJKLMNOP"), length + 4)
        chain = nodes[: length + 1]
        edges = list(zip(chain[:-1], chain[1:])) + [(nodes[-3], nodes[-2])]
        canonical = ",".join(f"{a}>{b}" for a, b in sorted(edges))
        identifier = semantic_id(task, f"{chain[0]}:{length}:{canonical}")
        rng.shuffle(edges)
        rules = ", ".join(f"{a}->{b}" for a, b in edges)
        question = f"Links: {rules}. Follow {length} links from {chain[0]}. Where do you end?"
        reasoning = " -> ".join(chain) + "."
        answer = chain[-1]
    else:
        raise ValueError(f"Unknown task: {task}")
    return Example(
        identifier, task, question, reasoning, answer, "ood" if ood else assigned_split(identifier)
    )


def generate_dataset(
    directory: Path, train: int, validation: int, test: int, ood: int, seed: int = 20260905
) -> dict:
    sizes = {"train": train, "validation": validation, "test": test, "ood": ood}
    if any(n < 0 or n % 2 for n in sizes.values()):
        raise ValueError("Split sizes must be nonnegative and even")
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = {split: [] for split in sizes}
    seen = set()
    for task in ("arithmetic", "links"):
        for split, size in sizes.items():
            collected = 0
            attempts = 0
            while collected < size // 2:
                attempts += 1
                if attempts > max(100_000, size * 1000):
                    raise ValueError("Requested split exceeds available unique task examples")
                example = make_candidate(rng, task, ood=split == "ood")
                if example.split != split or example.id in seen:
                    continue
                seen.add(example.id)
                records[split].append(example)
                collected += 1
    hashes = {}
    for split, examples in records.items():
        rng.shuffle(examples)
        body = "".join(json.dumps(asdict(x), sort_keys=True) + "\n" for x in examples)
        (directory / f"{split}.jsonl").write_text(body)
        hashes[split] = hashlib.sha256(body.encode()).hexdigest()
    manifest = {
        "seed": seed,
        "sizes": sizes,
        "sha256": hashes,
        "generator_version": 1,
        "tasks": ["arithmetic", "links"],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_examples(path: Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line]


def prompt_text(
    tokenizer, example: Example, native_thinking: bool = False, reasoning_prefix: bool = True
) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example.question + "\nGive the final result after Answer:."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=native_thinking,
    )
    return text if native_thinking or not reasoning_prefix else text + "Reasoning: "


def remaining_reasoning(example: Example, steps_to_drop: int) -> str:
    """Remove a prefix of annotated reasoning, without supplying it in the prompt."""
    if steps_to_drop < 0:
        raise ValueError("steps_to_drop must be nonnegative")
    if steps_to_drop == 0:
        return example.reasoning
    if example.task == "links":
        nodes = example.reasoning.rstrip(".").split(" -> ")
        return " -> ".join(nodes[steps_to_drop:]) + "." if steps_to_drop < len(nodes) - 1 else ""
    if example.task == "arithmetic":
        # Diagnostic arithmetic uses two integer equations separated by '. '.
        steps = example.reasoning.split(". ")
        return ". ".join(steps[steps_to_drop:])
    if example.task == "gsm8k":
        # The official worked solutions use newline-separated reasoning steps.
        steps = [line.strip() for line in example.reasoning.splitlines() if line.strip()]
        return "\n".join(steps[steps_to_drop:])
    raise ValueError(f"No reasoning-step annotation policy for {example.task}")


def encode_example(tokenizer, example: Example, mode: str, reasoning_steps_to_drop: int = 0):
    if mode not in {"direct", "cot", "latent", "hybrid"}:
        raise ValueError(f"Unknown mode: {mode}")
    prompt = tokenizer.encode(prompt_text(tokenizer, example), add_special_tokens=False)
    suffix = "\nAnswer: "
    if mode in {"cot", "hybrid"}:
        reasoning = (
            remaining_reasoning(example, reasoning_steps_to_drop)
            if mode == "hybrid"
            else example.reasoning
        )
        ids = tokenizer.encode(reasoning + suffix + example.answer, add_special_tokens=False)
        prefix_len = 0
    else:
        # Match the decoder's forced prefix exactly. Encoding the concatenated
        # string can merge its trailing space with the first answer token and
        # would incorrectly mask that token (notably for Gemma letter answers).
        prefix = tokenizer.encode(suffix, add_special_tokens=False)
        ids = prefix + tokenizer.encode(example.answer, add_special_tokens=False)
        prefix_len = len(prefix)
    ids.append(tokenizer.eos_token_id)
    mask = [0.0] * prefix_len + [1.0] * (len(ids) - prefix_len)
    return prompt, ids, mask


def extract_answer(text: str, task: str, mode: str) -> str | None:
    if mode == "native" and "<|channel>thought" in text:
        if "<channel|>" not in text:
            return None
        text = text.rsplit("<channel|>", 1)[-1]
    if mode in {"cot", "native", "hybrid", "plain"}:
        parts = re.split(r"Answer:\s*", text, flags=re.IGNORECASE)
        if len(parts) < 2 or not parts[-1].strip():
            return None
        text = parts[-1].splitlines()[0]
    text = text.strip().strip("*").strip()
    if task in {"arithmetic", "gsm8k"}:
        match = re.match(r"\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
        return match.group(1).replace(",", "").lstrip("+") if match else None
    if task == "links":
        match = re.fullmatch(r"([A-P])[.!]?", text)
        return match.group(1) if match else None
    raise ValueError(f"Unknown task: {task}")

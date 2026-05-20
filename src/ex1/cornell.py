from __future__ import annotations

import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from src import config
from src.ex1.dataset import DialogueExample
from src.ex1.text import tokenize


def _read_lines_file(path: Path) -> dict[str, str]:
    line_id_to_text: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split(" +++$+++ ")
            if len(parts) != 5:
                continue
            line_id_to_text[parts[0]] = parts[4]
    return line_id_to_text


def _read_conversations_file(path: Path) -> list[dict[str, object]]:
    conversations: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split(" +++$+++ ")
            if len(parts) != 4:
                continue
            conversations.append(
                {
                    "character1_id": parts[0],
                    "character2_id": parts[1],
                    "movie_id": parts[2],
                    "line_ids": list(ast.literal_eval(parts[3])),
                }
            )
    return conversations


def _is_useful_pair(source: str, target: str, min_tokens: int, max_tokens: int) -> bool:
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    if len(source_tokens) < min_tokens or len(target_tokens) < min_tokens:
        return False
    if len(source_tokens) > max_tokens or len(target_tokens) > max_tokens:
        return False
    if not source_tokens or not target_tokens:
        return False
    return True


def load_cornell_dialogue_pairs(
    corpus_dir: Path | None = None,
    *,
    min_tokens: int = 2,
    max_tokens: int = config.MAX_SOURCE_TOKENS,
    max_conversations: int | None = None,
) -> list[DialogueExample]:
    """Load consecutive utterance pairs from the Cornell Movie-Dialogs Corpus."""
    corpus_dir = corpus_dir or config.CORNELL_MOVIE_DIR
    line_map = _read_lines_file(corpus_dir / "movie_lines.txt")
    conversations = _read_conversations_file(corpus_dir / "movie_conversations.txt")
    examples: list[DialogueExample] = []

    for conversation_index, conversation in enumerate(conversations):
        if max_conversations is not None and conversation_index >= max_conversations:
            break
        line_ids = conversation["line_ids"]
        assert isinstance(line_ids, list)
        for source_id, target_id in zip(line_ids, line_ids[1:]):
            source_text = line_map.get(source_id)
            target_text = line_map.get(target_id)
            if source_text is None or target_text is None:
                continue
            if not _is_useful_pair(source_text, target_text, min_tokens, max_tokens):
                continue
            examples.append(
                DialogueExample(
                    source_text=source_text,
                    target_text=target_text,
                    conversation_id=f"movie_{conversation_index}",
                    source_line_id=source_id,
                    target_line_id=target_id,
                    metadata={
                        "movie_id": str(conversation["movie_id"]),
                        "character1_id": str(conversation["character1_id"]),
                        "character2_id": str(conversation["character2_id"]),
                    },
                )
            )
    return examples


def split_by_conversation(
    examples: Iterable[DialogueExample],
    *,
    seed: int = config.RANDOM_SEED,
    train_ratio: float = config.TRAIN_RATIO,
    valid_ratio: float = config.VALID_RATIO,
    test_ratio: float = config.TEST_RATIO,
) -> tuple[list[DialogueExample], list[DialogueExample], list[DialogueExample]]:
    """Split examples by conversation id to reduce leakage across splits."""
    examples = list(examples)
    convo_to_examples: dict[str, list[DialogueExample]] = {}
    for example in examples:
        convo_to_examples.setdefault(example.conversation_id, []).append(example)

    convo_ids = list(convo_to_examples)
    random.Random(seed).shuffle(convo_ids)

    total = len(convo_ids)
    train_cut = int(total * train_ratio)
    valid_cut = train_cut + int(total * valid_ratio)
    train_ids = set(convo_ids[:train_cut])
    valid_ids = set(convo_ids[train_cut:valid_cut])
    test_ids = set(convo_ids[valid_cut:])

    train_examples: list[DialogueExample] = []
    valid_examples: list[DialogueExample] = []
    test_examples: list[DialogueExample] = []

    for convo_id, convo_examples in convo_to_examples.items():
        if convo_id in train_ids:
            train_examples.extend(convo_examples)
        elif convo_id in valid_ids:
            valid_examples.extend(convo_examples)
        else:
            test_examples.extend(convo_examples)

    return train_examples, valid_examples, test_examples


def compute_eda_summary(examples: Iterable[DialogueExample]) -> dict[str, object]:
    examples = list(examples)
    conversation_ids = {example.conversation_id for example in examples}
    source_lengths = [len(tokenize(example.source_text)) for example in examples]
    target_lengths = [len(tokenize(example.target_text)) for example in examples]

    token_counter = Counter()
    for example in examples:
        token_counter.update(tokenize(example.source_text))
        token_counter.update(tokenize(example.target_text))

    noisy_examples = []
    for example in examples:
        source_tokens = tokenize(example.source_text)
        target_tokens = tokenize(example.target_text)
        has_repeated_punctuation = any(
            pattern in example.source_text for pattern in ("...", "--", "??", "!!")
        ) or any(
            pattern in example.target_text for pattern in ("...", "--", "??", "!!")
        )
        if len(source_tokens) <= 2 or len(target_tokens) <= 2:
            noisy_examples.append(
                {
                    "source": example.source_text,
                    "target": example.target_text,
                    "reason": "very_short",
                }
            )
        elif has_repeated_punctuation:
            noisy_examples.append(
                {
                    "source": example.source_text,
                    "target": example.target_text,
                    "reason": "punctuation_heavy",
                }
            )
        if len(noisy_examples) >= 5:
            break

    return {
        "num_pairs": len(examples),
        "num_conversations": len(conversation_ids),
        "vocab_size": len(token_counter),
        "avg_source_length": round(
            sum(source_lengths) / max(1, len(source_lengths)), 2
        ),
        "avg_target_length": round(
            sum(target_lengths) / max(1, len(target_lengths)), 2
        ),
        "median_source_length": (
            int(sorted(source_lengths)[len(source_lengths) // 2])
            if source_lengths
            else 0
        ),
        "median_target_length": (
            int(sorted(target_lengths)[len(target_lengths) // 2])
            if target_lengths
            else 0
        ),
        "top_tokens": token_counter.most_common(20),
        "noisy_examples": noisy_examples,
    }


def save_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_examples(examples: Iterable[DialogueExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.__dict__, ensure_ascii=False) + "\n")

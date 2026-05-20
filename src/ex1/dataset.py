from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from src import config
from src.ex1.text import tokenize


@dataclass(frozen=True)
class DialogueExample:
    source_text: str
    target_text: str
    conversation_id: str
    source_line_id: str
    target_line_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DialogueDataset(Dataset):
    def __init__(
        self, examples: list[DialogueExample], vocabulary: "Vocabulary"
    ) -> None:
        self.examples = examples
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        source_tokens = tokenize(example.source_text)[: config.MAX_SOURCE_TOKENS]
        target_tokens = tokenize(example.target_text)[: config.MAX_TARGET_TOKENS]

        source_ids = self.vocabulary.encode(source_tokens, add_eos=True)
        target_input_ids = self.vocabulary.encode(target_tokens, add_bos=True)
        target_output_ids = self.vocabulary.encode(target_tokens, add_eos=True)

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_input_ids": torch.tensor(target_input_ids, dtype=torch.long),
            "target_output_ids": torch.tensor(target_output_ids, dtype=torch.long),
            "source_length": torch.tensor(len(source_ids), dtype=torch.long),
            "target_length": torch.tensor(len(target_output_ids), dtype=torch.long),
        }


def pad_sequences(sequences: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_length = max(sequence.size(0) for sequence in sequences)
    batch = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        batch[row_index, : sequence.size(0)] = sequence
    return batch


def collate_dialogue_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    source_ids = [item["source_ids"] for item in batch]
    target_input_ids = [item["target_input_ids"] for item in batch]
    target_output_ids = [item["target_output_ids"] for item in batch]

    return {
        "source_ids": pad_sequences(source_ids, config.PAD_ID),
        "target_input_ids": pad_sequences(target_input_ids, config.PAD_ID),
        "target_output_ids": pad_sequences(target_output_ids, config.PAD_ID),
        "source_lengths": torch.stack([item["source_length"] for item in batch]),
        "target_lengths": torch.stack([item["target_length"] for item in batch]),
    }

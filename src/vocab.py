from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.text import tokenize


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[config.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[config.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[config.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[config.EOS_TOKEN]

    @property
    def id_to_token(self) -> list[str]:
        return [
            token
            for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1])
        ]

    def encode(
        self, tokens: list[str], *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(token, self.unk_id) for token in tokens)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, stop_at_eos: bool = True) -> list[str]:
        reverse = self.id_to_token
        tokens: list[str] = []
        for identifier in ids:
            token = (
                reverse[identifier]
                if 0 <= identifier < len(reverse)
                else config.UNK_TOKEN
            )
            if stop_at_eos and token == config.EOS_TOKEN:
                break
            if token in {config.BOS_TOKEN, config.PAD_TOKEN}:
                continue
            tokens.append(token)
        return tokens

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for token in self.id_to_token:
                handle.write(token + "\n")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with path.open("r", encoding="utf-8") as handle:
            tokens = [line.rstrip("\n") for line in handle]
        return cls({token: index for index, token in enumerate(tokens)})


def build_vocabulary(
    examples,
    *,
    min_freq: int = config.MIN_TOKEN_FREQ,
    max_size: int = config.MAX_VOCAB_SIZE,
) -> Vocabulary:
    counter = Counter()
    for example in examples:
        counter.update(tokenize(example.source_text))
        counter.update(tokenize(example.target_text))

    token_to_id = {
        config.PAD_TOKEN: config.PAD_ID,
        config.UNK_TOKEN: config.UNK_ID,
        config.BOS_TOKEN: config.BOS_ID,
        config.EOS_TOKEN: config.EOS_ID,
    }
    for token, frequency in counter.most_common():
        if frequency < min_freq:
            continue
        if token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)
        if len(token_to_id) >= max_size:
            break
    return Vocabulary(token_to_id)

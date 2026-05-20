from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from src import config


class Encoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_ID)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, source_ids: torch.Tensor, source_lengths: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.dropout(self.embedding(source_ids))
        packed = pack_padded_sequence(
            embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden


class Decoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_ID)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self, input_tokens: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens)).unsqueeze(1)
        output, hidden = self.gru(embedded, hidden)
        logits = self.output(output.squeeze(1))
        return logits, hidden


@dataclass
class GenerationResult:
    token_ids: list[int]
    log_probability: float


class Seq2SeqChatbot(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size, config.EMBED_DIM, config.HIDDEN_DIM, config.DROPOUT
        )
        self.decoder = Decoder(
            vocab_size, config.EMBED_DIM, config.HIDDEN_DIM, config.DROPOUT
        )
        self.vocab_size = vocab_size

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        teacher_forcing_ratio: float = config.TEACHER_FORCING_RATIO,
    ) -> torch.Tensor:
        hidden = self.encoder(source_ids, source_lengths)
        input_tokens = target_input_ids[:, 0]
        outputs = []
        for step in range(target_input_ids.size(1)):
            logits, hidden = self.decoder.forward_step(input_tokens, hidden)
            outputs.append(logits)
            if step + 1 >= target_input_ids.size(1):
                break
            if random.random() < teacher_forcing_ratio:
                input_tokens = target_input_ids[:, step + 1]
            else:
                input_tokens = logits.argmax(dim=-1)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        source_length: torch.Tensor,
        *,
        max_length: int,
        bos_id: int,
        eos_id: int,
    ) -> GenerationResult:
        self.eval()
        hidden = self.encoder(source_ids, source_length)
        input_token = torch.tensor([bos_id], device=source_ids.device)
        generated: list[int] = []
        log_probability = 0.0
        for _ in range(max_length):
            logits, hidden = self.decoder.forward_step(input_token, hidden)
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = int(torch.argmax(log_probs, dim=-1).item())
            log_probability += float(log_probs[0, next_token].item())
            if next_token == eos_id:
                break
            generated.append(next_token)
            input_token = torch.tensor([next_token], device=source_ids.device)
        return GenerationResult(generated, log_probability)

    @torch.no_grad()
    def beam_search(
        self,
        source_ids: torch.Tensor,
        source_length: torch.Tensor,
        *,
        max_length: int,
        bos_id: int,
        eos_id: int,
        beam_size: int = 5,
        length_penalty: float = 0.7,
    ) -> GenerationResult:
        self.eval()
        hidden = self.encoder(source_ids, source_length)
        beams: list[tuple[list[int], torch.Tensor, float]] = [([bos_id], hidden, 0.0)]

        def score_for(sequence: list[int], log_prob: float) -> float:
            length = max(1, len(sequence) - 1)
            return log_prob / (length**length_penalty)

        for _ in range(max_length):
            candidates: list[tuple[list[int], torch.Tensor, float]] = []
            all_finished = True
            for sequence, beam_hidden, beam_log_prob in beams:
                last_token = sequence[-1]
                if last_token == eos_id:
                    candidates.append((sequence, beam_hidden, beam_log_prob))
                    continue
                all_finished = False
                input_token = torch.tensor([last_token], device=source_ids.device)
                logits, next_hidden = self.decoder.forward_step(
                    input_token, beam_hidden
                )
                log_probs = F.log_softmax(logits, dim=-1)[0]
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                for token_score, token_id in zip(
                    top_log_probs.tolist(), top_indices.tolist()
                ):
                    candidates.append(
                        (
                            sequence + [token_id],
                            next_hidden.clone(),
                            beam_log_prob + token_score,
                        )
                    )
            if all_finished:
                break
            candidates.sort(key=lambda item: score_for(item[0], item[2]), reverse=True)
            beams = candidates[:beam_size]

        best_sequence, _, best_log_prob = max(
            beams, key=lambda item: score_for(item[0], item[2])
        )
        cleaned_sequence = [token for token in best_sequence[1:] if token != eos_id]
        return GenerationResult(cleaned_sequence, best_log_prob)


def sequence_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, pad_id: int
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id
    )


def perplexity_from_loss(loss_value: float) -> float:
    return math.exp(min(loss_value, 20.0))

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import config


def load_glove_embeddings(
    glove_path: Path,
    token_to_id: dict[str, int],
    embedding_dim: int,
) -> tuple[torch.Tensor, int]:
    """Build an embedding matrix initialized from a GloVe text file."""
    matrix = torch.empty(len(token_to_id), embedding_dim)
    nn.init.normal_(matrix, mean=0.0, std=0.05)
    if 0 <= config.PAD_ID < len(token_to_id):
        matrix[config.PAD_ID].zero_()

    matched = 0
    with glove_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != embedding_dim + 1:
                continue
            token = parts[0]
            token_id = token_to_id.get(token)
            if token_id is None:
                continue
            values = [float(value) for value in parts[1:]]
            matrix[token_id] = torch.tensor(values, dtype=torch.float32)
            matched += 1
    return matrix, matched


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_ID)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def forward(
        self, source_ids: torch.Tensor, source_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source_ids))
        packed = pack_padded_sequence(
            embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, decoder_hidden_dim: int, encoder_output_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.key = nn.Linear(encoder_output_dim, decoder_hidden_dim, bias=False)
        self.energy = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(decoder_state).unsqueeze(1)
        key = self.key(encoder_outputs)
        scores = self.energy(torch.tanh(query + key)).squeeze(-1)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        decoder_hidden_dim: int,
        encoder_output_dim: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_ID)
        self.attention = AdditiveAttention(decoder_hidden_dim, encoder_output_dim)
        self.gru = nn.GRU(
            embed_dim + encoder_output_dim,
            decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(decoder_hidden_dim + encoder_output_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens))
        context, _ = self.attention(hidden[-1], encoder_outputs, source_mask)
        step_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        output, hidden = self.gru(step_input, hidden)
        logits = self.output(torch.cat([output.squeeze(1), context], dim=-1))
        return logits, hidden


@dataclass
class GenerationResult:
    token_ids: list[int]
    log_probability: float


class Seq2SeqChatbot(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        num_layers: int = 2,
        bidirectional_encoder: bool = True,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            dropout,
            num_layers=num_layers,
            bidirectional=bidirectional_encoder,
        )

        encoder_output_dim = hidden_dim * (2 if bidirectional_encoder else 1)
        self.decoder = AttentionDecoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            encoder_output_dim,
            dropout,
            num_layers=num_layers,
        )
        self.hidden_bridge = nn.Linear(encoder_output_dim, hidden_dim)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder

        if pretrained_embeddings is not None:
            if pretrained_embeddings.size(1) != embed_dim:
                raise ValueError(
                    "Embedding dimension mismatch between model and pretrained matrix."
                )
            self.encoder.embedding.weight.data.copy_(pretrained_embeddings)
            self.decoder.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.encoder.embedding.weight.requires_grad = False
                self.decoder.embedding.weight.requires_grad = False

    def _source_mask(
        self, source_ids: torch.Tensor, source_lengths: torch.Tensor
    ) -> torch.Tensor:
        max_len = source_ids.size(1)
        device = source_ids.device
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        return positions < source_lengths.unsqueeze(1)

    def _init_decoder_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        if self.bidirectional_encoder:
            # Merge forward/backward states from the top encoder layer and broadcast to all decoder layers.
            forward = encoder_hidden[-2]
            backward = encoder_hidden[-1]
            merged = torch.cat([forward, backward], dim=-1)
        else:
            merged = encoder_hidden[-1]
        initial_state = torch.tanh(self.hidden_bridge(merged))
        return initial_state.unsqueeze(0).repeat(self.num_layers, 1, 1)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        teacher_forcing_ratio: float = config.TEACHER_FORCING_RATIO,
    ) -> torch.Tensor:
        encoder_outputs, encoder_hidden = self.encoder(source_ids, source_lengths)
        hidden = self._init_decoder_hidden(encoder_hidden)
        source_mask = self._source_mask(source_ids, source_lengths)

        input_tokens = target_input_ids[:, 0]
        outputs = []
        for step in range(target_input_ids.size(1)):
            logits, hidden = self.decoder.forward_step(
                input_tokens, hidden, encoder_outputs, source_mask
            )
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
        encoder_outputs, encoder_hidden = self.encoder(source_ids, source_length)
        hidden = self._init_decoder_hidden(encoder_hidden)
        source_mask = self._source_mask(source_ids, source_length)

        input_token = torch.tensor([bos_id], device=source_ids.device)
        generated: list[int] = []
        log_probability = 0.0
        for _ in range(max_length):
            logits, hidden = self.decoder.forward_step(
                input_token, hidden, encoder_outputs, source_mask
            )
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = int(torch.argmax(log_probs, dim=-1).item())
            log_probability += float(log_probs[0, next_token].item())
            if next_token == eos_id:
                break
            generated.append(next_token)
            input_token = torch.tensor([next_token], device=source_ids.device)
        return GenerationResult(generated, log_probability)

    @torch.no_grad()
    def sample_decode(
        self,
        source_ids: torch.Tensor,
        source_length: torch.Tensor,
        *,
        max_length: int,
        bos_id: int,
        eos_id: int,
        top_k: int = 20,
        temperature: float = 0.9,
    ) -> GenerationResult:
        self.eval()
        encoder_outputs, encoder_hidden = self.encoder(source_ids, source_length)
        hidden = self._init_decoder_hidden(encoder_hidden)
        source_mask = self._source_mask(source_ids, source_length)

        input_token = torch.tensor([bos_id], device=source_ids.device)
        generated: list[int] = []
        log_probability = 0.0
        temperature = max(temperature, 1e-4)

        for _ in range(max_length):
            logits, hidden = self.decoder.forward_step(
                input_token, hidden, encoder_outputs, source_mask
            )
            scaled_logits = logits / temperature
            if top_k > 0:
                top_values, top_indices = torch.topk(
                    scaled_logits, k=min(top_k, scaled_logits.size(-1)), dim=-1
                )
                filtered_logits = torch.full_like(scaled_logits, float("-inf"))
                filtered_logits.scatter_(1, top_indices, top_values)
                scaled_logits = filtered_logits

            probs = F.softmax(scaled_logits, dim=-1)
            sampled = int(torch.multinomial(probs, num_samples=1).item())
            log_probability += float(torch.log(probs[0, sampled] + 1e-12).item())
            if sampled == eos_id:
                break
            generated.append(sampled)
            input_token = torch.tensor([sampled], device=source_ids.device)
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
        encoder_outputs, encoder_hidden = self.encoder(source_ids, source_length)
        source_mask = self._source_mask(source_ids, source_length)
        initial_hidden = self._init_decoder_hidden(encoder_hidden)

        beams: list[tuple[list[int], torch.Tensor, float]] = [
            ([bos_id], initial_hidden, 0.0)
        ]

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
                    input_token, beam_hidden, encoder_outputs, source_mask
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

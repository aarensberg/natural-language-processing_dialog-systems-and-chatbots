from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.ex1.cornell import (
    compute_eda_summary,
    load_cornell_dialogue_pairs,
    save_examples,
    save_json,
    split_by_conversation,
)
from src.ex1.dataset import DialogueDataset, collate_dialogue_batch
from src.ex1.model import Seq2SeqChatbot, perplexity_from_loss, sequence_cross_entropy
from src.ex1.text import detokenize, ngram_diversity, tokenize
from src.ex1.vocab import Vocabulary, build_vocabulary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(
    train_examples, valid_examples, test_examples, vocabulary, batch_size: int
):
    train_dataset = DialogueDataset(train_examples, vocabulary)
    valid_dataset = DialogueDataset(valid_examples, vocabulary)
    test_dataset = DialogueDataset(test_examples, vocabulary)

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_dialogue_batch,
        ),
        DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dialogue_batch,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dialogue_batch,
        ),
    )


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        source_ids = batch["source_ids"].to(device)
        source_lengths = batch["source_lengths"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_output_ids = batch["target_output_ids"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(source_ids, source_lengths, target_input_ids)
        loss = sequence_cross_entropy(logits, target_output_ids, config.PAD_ID)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in tqdm(loader, desc="valid", leave=False):
        source_ids = batch["source_ids"].to(device)
        source_lengths = batch["source_lengths"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_output_ids = batch["target_output_ids"].to(device)
        logits = model(
            source_ids, source_lengths, target_input_ids, teacher_forcing_ratio=1.0
        )
        loss = sequence_cross_entropy(logits, target_output_ids, config.PAD_ID)
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)


def tensorize_example(
    example_text: str, vocabulary: Vocabulary, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    source_tokens = tokenize(example_text)[: config.MAX_SOURCE_TOKENS]
    source_ids = vocabulary.encode(source_tokens, add_eos=True)
    return torch.tensor([source_ids], dtype=torch.long, device=device), torch.tensor(
        [len(source_ids)], dtype=torch.long, device=device
    )


def decode_ids(vocabulary: Vocabulary, token_ids: list[int]) -> str:
    return detokenize(vocabulary.decode(token_ids, stop_at_eos=True))


@torch.no_grad()
def generate_samples(
    model, vocabulary, device, examples, output_path: Path, num_examples: int = 10
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_examples = random.sample(examples, k=min(num_examples, len(examples)))
    with output_path.open("w", encoding="utf-8") as handle:
        for example in chosen_examples:
            source_ids, source_length = tensorize_example(
                example.source_text, vocabulary, device
            )
            greedy = model.greedy_decode(
                source_ids,
                source_length,
                max_length=config.MAX_TARGET_TOKENS,
                bos_id=vocabulary.bos_id,
                eos_id=vocabulary.eos_id,
            )
            beam = model.beam_search(
                source_ids,
                source_length,
                max_length=config.MAX_TARGET_TOKENS,
                bos_id=vocabulary.bos_id,
                eos_id=vocabulary.eos_id,
            )
            handle.write(f"USER: {example.source_text}\n")
            handle.write(f"REFERENCE: {example.target_text}\n")
            handle.write(f"GREEDY: {decode_ids(vocabulary, greedy.token_ids)}\n")
            handle.write(f"BEAM: {decode_ids(vocabulary, beam.token_ids)}\n")
            handle.write("-" * 72 + "\n")


def write_metrics(
    summary: dict[str, object],
    train_loss: float,
    valid_loss: float,
    test_loss: float,
    output_path: Path,
) -> None:
    payload = dict(summary)
    payload.update(
        {
            "train_loss": round(train_loss, 4),
            "valid_loss": round(valid_loss, 4),
            "test_loss": round(test_loss, 4),
            "train_perplexity": round(perplexity_from_loss(train_loss), 4),
            "valid_perplexity": round(perplexity_from_loss(valid_loss), 4),
            "test_perplexity": round(perplexity_from_loss(test_loss), 4),
        }
    )
    save_json(payload, output_path)


def save_eda_plots(
    source_lengths: list[int],
    target_lengths: list[int],
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    top_tokens = summary.get("top_tokens", [])

    if source_lengths and target_lengths:
        plt.figure(figsize=(8, 4))
        plt.hist(source_lengths, bins=20, alpha=0.6, label="source")
        plt.hist(target_lengths, bins=20, alpha=0.6, label="target")
        plt.xlabel("Token length")
        plt.ylabel("Count")
        plt.title("Cornell dialogue length distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "length_histogram.png", dpi=160)
        plt.close()

    if top_tokens:
        labels = [token for token, _ in top_tokens[:15]]
        values = [count for _, count in top_tokens[:15]]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title("Most common tokens in Cornell pairs")
        plt.tight_layout()
        plt.savefig(output_dir / "top_tokens.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise 1 pipeline: Cornell preprocessing, EDA, and baseline chatbot training."
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Optional cap for faster smoke tests.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=config.EX1_DIR)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.EX1_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.EX1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.EX1_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_cornell_dialogue_pairs(max_conversations=args.max_conversations)
    train_examples, valid_examples, test_examples = split_by_conversation(examples)

    split_manifest = {
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "test_examples": len(test_examples),
    }
    save_json(split_manifest, args.output_dir / "split_manifest.json")
    save_examples(
        train_examples[:200], args.output_dir / "train_examples_preview.jsonl"
    )

    eda_summary = compute_eda_summary(examples)
    save_json(eda_summary, args.output_dir / "eda_summary.json")
    save_eda_plots(
        [len(tokenize(example.source_text)) for example in examples],
        [len(tokenize(example.target_text)) for example in examples],
        eda_summary,
        args.output_dir,
    )

    vocabulary = build_vocabulary(train_examples)
    vocabulary.save(args.output_dir / "vocab.txt")
    save_json(
        {"vocab_size": len(vocabulary.token_to_id)},
        args.output_dir / "vocab_summary.json",
    )

    train_loader, valid_loader, test_loader = make_dataloaders(
        train_examples, valid_examples, test_examples, vocabulary, args.batch_size
    )

    model = Seq2SeqChatbot(vocab_size=len(vocabulary.token_to_id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_valid = math.inf
    checkpoint_path = config.EX1_CHECKPOINT_DIR / "baseline_seq2seq.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = evaluate(model, valid_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocabulary.token_to_id,
                },
                checkpoint_path,
            )

    test_loss = evaluate(model, test_loader, device)
    save_json({"history": history}, args.output_dir / "training_history.json")
    write_metrics(
        eda_summary,
        history[-1]["train_loss"],
        history[-1]["valid_loss"],
        test_loss,
        args.output_dir / "metrics.json",
    )

    generate_samples(
        model,
        vocabulary,
        device,
        test_examples,
        args.output_dir / "sample_conversations.txt",
    )

    distinct_1 = ngram_diversity(
        (tokenize(example.target_text) for example in test_examples), 1
    )
    distinct_2 = ngram_diversity(
        (tokenize(example.target_text) for example in test_examples), 2
    )
    save_json(
        {
            "distinct_1_reference": round(distinct_1, 4),
            "distinct_2_reference": round(distinct_2, 4),
        },
        args.output_dir / "reference_diversity.json",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.cornell import (
    compute_eda_summary,
    load_cornell_dialogue_pairs,
    save_examples,
    save_json,
    split_by_conversation,
)
from src.dataset import DialogueDataset, DialogueExample, collate_dialogue_batch
from src.model import Seq2SeqChatbot
from src.pipeline import (
    decode_ids,
    evaluate,
    evaluate_generated_outputs,
    maybe_load_glove_matrix,
    make_dataloaders,
    ngram_diversity,
    set_seed,
    tensorize_example,
    train_one_epoch,
    write_metrics,
)
from src.text import detokenize, tokenize
from src.vocab import Vocabulary, build_vocabulary

SUPPORTED_DOMAINS = {"cornell", "personachat"}
EXPERIMENTS = {
    "cornell_only": ["cornell"],
    "cornell_plus_persona": ["cornell", "personachat"],
}


def prefix_examples(
    examples: list[DialogueExample], domain_name: str
) -> list[DialogueExample]:
    prefixed_examples: list[DialogueExample] = []
    for example in examples:
        prefixed_examples.append(
            DialogueExample(
                source_text=f"corpus {domain_name} {example.source_text}",
                target_text=example.target_text,
                conversation_id=f"{domain_name}_{example.conversation_id}",
                source_line_id=example.source_line_id,
                target_line_id=example.target_line_id,
                metadata={**example.metadata, "domain": domain_name},
            )
        )
    return prefixed_examples


def strip_domain_prefix(text: str) -> str:
    tokens = tokenize(text)
    if len(tokens) >= 2 and tokens[0] == "corpus" and tokens[1] in SUPPORTED_DOMAINS:
        return detokenize(tokens[2:])
    return text


def _is_useful_pair(source: str, target: str, min_tokens: int, max_tokens: int) -> bool:
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    if len(source_tokens) < min_tokens or len(target_tokens) < min_tokens:
        return False
    if len(source_tokens) > max_tokens or len(target_tokens) > max_tokens:
        return False
    return bool(source_tokens and target_tokens)


def load_personachat_dialogue_pairs(
    corpus_dir: Path | None = None,
    *,
    split_name: str = "train",
    variant: str = "self_original",
    max_dialogues: int | None = None,
    min_tokens: int = 2,
    max_tokens: int = config.MAX_SOURCE_TOKENS,
) -> list[DialogueExample]:
    corpus_dir = corpus_dir or config.PERSONA_CHAT_DIR
    file_path = corpus_dir / f"{split_name}_{variant}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"PersonaChat file not found: {file_path}")

    examples: list[DialogueExample] = []
    conversation_index = -1
    current_persona: list[str] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            if line.startswith("1 your persona:"):
                conversation_index += 1
                current_persona = [line.split("your persona:", 1)[1].strip()]
                continue

            if "your persona:" in line and "\t" not in line:
                current_persona.append(line.split("your persona:", 1)[1].strip())
                continue

            if "\t" not in line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            source_field = parts[0].strip()
            target_text = parts[1].strip()
            if not source_field or not target_text:
                continue

            if " " in source_field:
                turn_id, source_text = source_field.split(" ", 1)
            else:
                turn_id, source_text = "0", source_field

            if conversation_index < 0:
                conversation_index = 0

            if not _is_useful_pair(source_text, target_text, min_tokens, max_tokens):
                continue

            examples.append(
                DialogueExample(
                    source_text=source_text,
                    target_text=target_text,
                    conversation_id=f"personachat_{split_name}_{conversation_index}",
                    source_line_id=f"{split_name}_{conversation_index}_{turn_id}_src",
                    target_line_id=f"{split_name}_{conversation_index}_{turn_id}_tgt",
                    metadata={
                        "dataset": "personachat",
                        "split": split_name,
                        "variant": variant,
                        "persona": current_persona[:],
                        "turn_id": turn_id,
                    },
                )
            )

            if (
                max_dialogues is not None
                and len({example.conversation_id for example in examples})
                >= max_dialogues
            ):
                break

    return examples


def make_loader(
    examples: list[DialogueExample],
    vocabulary: Vocabulary,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = DialogueDataset(examples, vocabulary)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dialogue_batch,
    )


def save_length_plot(
    lengths_by_domain: dict[str, list[int]],
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    for domain_name, lengths in lengths_by_domain.items():
        if lengths:
            plt.hist(lengths, bins=20, alpha=0.55, label=domain_name)
    plt.xlabel("Token length")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_top_tokens_plot(
    summary: dict[str, object], output_path: Path, *, title: str
) -> None:
    top_tokens = summary.get("top_tokens", [])
    if not top_tokens:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [token for token, _ in top_tokens[:15]]
    values = [count for _, count in top_tokens[:15]]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


@torch.no_grad()
def write_sample_conversations(
    model: Seq2SeqChatbot,
    vocabulary: Vocabulary,
    device: torch.device,
    examples: list[DialogueExample],
    output_path: Path,
    *,
    num_examples: int = 10,
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
            topk_temp = model.sample_decode(
                source_ids,
                source_length,
                max_length=config.MAX_TARGET_TOKENS,
                bos_id=vocabulary.bos_id,
                eos_id=vocabulary.eos_id,
                top_k=30,
                temperature=0.8,
            )

            handle.write(f"DOMAIN: {example.metadata.get('domain', 'unknown')}\n")
            handle.write(f"USER: {strip_domain_prefix(example.source_text)}\n")
            handle.write(f"REFERENCE: {example.target_text}\n")
            handle.write(f"GREEDY: {decode_ids(vocabulary, greedy.token_ids)}\n")
            handle.write(f"BEAM: {decode_ids(vocabulary, beam.token_ids)}\n")
            handle.write(f"TOPK_TEMP: {decode_ids(vocabulary, topk_temp.token_ids)}\n")
            handle.write("-" * 72 + "\n")


def _count_conversations(examples: list[DialogueExample]) -> int:
    return len({example.conversation_id for example in examples})


def _concat_examples(groups: dict[str, list[DialogueExample]]) -> list[DialogueExample]:
    combined: list[DialogueExample] = []
    for domain_examples in groups.values():
        combined.extend(domain_examples)
    return combined


def _load_domain_splits(
    domain_name: str,
    *,
    max_cornell_conversations: int | None,
    max_persona_dialogues: int | None,
) -> tuple[list[DialogueExample], list[DialogueExample], list[DialogueExample]]:
    if domain_name == "cornell":
        examples = load_cornell_dialogue_pairs(
            max_conversations=max_cornell_conversations
        )
    elif domain_name == "personachat":
        examples = load_personachat_dialogue_pairs(max_dialogues=max_persona_dialogues)
    else:
        raise ValueError(f"Unsupported domain: {domain_name}")

    train_examples, valid_examples, test_examples = split_by_conversation(examples)
    return (
        prefix_examples(train_examples, domain_name),
        prefix_examples(valid_examples, domain_name),
        prefix_examples(test_examples, domain_name),
    )


def _evaluate_domain(
    model: Seq2SeqChatbot,
    vocabulary: Vocabulary,
    device: torch.device,
    examples: list[DialogueExample],
    batch_size: int,
) -> dict[str, object]:
    loader = make_loader(examples, vocabulary, batch_size, shuffle=False)
    loss = evaluate(model, loader, device)
    generation_metrics = evaluate_generated_outputs(model, vocabulary, device, examples)
    return {
        "loss": round(loss, 4),
        "perplexity": round(math.exp(loss), 4),
        "generation_metrics": generation_metrics,
        "reference_diversity": {
            "distinct_1": round(
                ngram_diversity(
                    (tokenize(example.target_text) for example in examples), 1
                ),
                4,
            ),
            "distinct_2": round(
                ngram_diversity(
                    (tokenize(example.target_text) for example in examples), 2
                ),
                4,
            ),
        },
        "num_examples": len(examples),
        "num_conversations": _count_conversations(examples),
    }


def run_experiment(
    experiment_name: str,
    dataset_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    output_dir = args.output_dir / experiment_name
    checkpoint_dir = config.EX3_CHECKPOINT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    domain_splits: dict[
        str, tuple[list[DialogueExample], list[DialogueExample], list[DialogueExample]]
    ] = {}
    for domain_name in dataset_names:
        domain_splits[domain_name] = _load_domain_splits(
            domain_name,
            max_cornell_conversations=args.max_conversations,
            max_persona_dialogues=args.max_persona_dialogues,
        )

    train_groups = {
        domain_name: splits[0] for domain_name, splits in domain_splits.items()
    }
    valid_groups = {
        domain_name: splits[1] for domain_name, splits in domain_splits.items()
    }
    test_groups = {
        domain_name: splits[2] for domain_name, splits in domain_splits.items()
    }

    train_examples = _concat_examples(train_groups)
    valid_examples = _concat_examples(valid_groups)
    test_examples = _concat_examples(test_groups)

    split_manifest = {
        "experiment": experiment_name,
        "datasets": dataset_names,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "test_examples": len(test_examples),
        "train_conversations": _count_conversations(train_examples),
        "valid_conversations": _count_conversations(valid_examples),
        "test_conversations": _count_conversations(test_examples),
        "by_domain": {
            domain_name: {
                "train_examples": len(splits[0]),
                "valid_examples": len(splits[1]),
                "test_examples": len(splits[2]),
                "train_conversations": _count_conversations(splits[0]),
                "valid_conversations": _count_conversations(splits[1]),
                "test_conversations": _count_conversations(splits[2]),
            }
            for domain_name, splits in domain_splits.items()
        },
    }
    save_json(split_manifest, output_dir / "split_manifest.json")

    for domain_name, splits in domain_splits.items():
        train_domain_examples, valid_domain_examples, test_domain_examples = splits
        save_json(
            {
                "train": compute_eda_summary(train_domain_examples),
                "valid": compute_eda_summary(valid_domain_examples),
                "test": compute_eda_summary(test_domain_examples),
            },
            output_dir / f"eda_summary_{domain_name}.json",
        )
        save_examples(
            train_domain_examples[:200],
            output_dir / f"train_examples_preview_{domain_name}.jsonl",
        )

    combined_summary = compute_eda_summary(
        train_examples + valid_examples + test_examples
    )
    save_json(combined_summary, output_dir / "eda_summary.json")
    save_examples(train_examples[:200], output_dir / "train_examples_preview.jsonl")

    save_length_plot(
        {
            domain_name: [len(tokenize(example.source_text)) for example in splits[0]]
            for domain_name, splits in domain_splits.items()
        },
        output_dir / "train_source_length_comparison.png",
        title=f"Training source length distribution - {experiment_name}",
    )
    save_length_plot(
        {
            domain_name: [len(tokenize(example.target_text)) for example in splits[0]]
            for domain_name, splits in domain_splits.items()
        },
        output_dir / "train_target_length_comparison.png",
        title=f"Training target length distribution - {experiment_name}",
    )
    save_top_tokens_plot(
        combined_summary,
        output_dir / "top_tokens.png",
        title=f"Most common tokens - {experiment_name}",
    )

    vocabulary = build_vocabulary(train_examples)
    vocabulary.save(output_dir / "vocab.txt")
    save_json(
        {"vocab_size": len(vocabulary.token_to_id)}, output_dir / "vocab_summary.json"
    )

    train_loader, valid_loader, test_loader = make_dataloaders(
        train_examples, valid_examples, test_examples, vocabulary, args.batch_size
    )

    pretrained_embeddings = None
    model_embed_dim = config.EMBED_DIM
    glove_matches = 0
    if args.use_glove:
        model_embed_dim = args.glove_dim
        pretrained_embeddings, glove_matches = maybe_load_glove_matrix(
            vocabulary,
            args.glove_path,
            args.glove_dim,
        )

    model = Seq2SeqChatbot(
        vocab_size=len(vocabulary.token_to_id),
        embed_dim=model_embed_dim,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        num_layers=args.layers,
        bidirectional_encoder=config.USE_BIDIRECTIONAL_ENCODER,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=args.freeze_glove,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_valid = math.inf
    checkpoint_path = checkpoint_dir / "seq2seq_attention.pt"

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
                    "embed_dim": model_embed_dim,
                    "num_layers": args.layers,
                    "glove_matches": glove_matches,
                    "experiment": experiment_name,
                    "datasets": dataset_names,
                },
                checkpoint_path,
            )

    test_loss = evaluate(model, test_loader, device)
    generation_metrics = evaluate_generated_outputs(
        model, vocabulary, device, test_examples
    )
    domain_metrics = {
        domain_name: _evaluate_domain(
            model,
            vocabulary,
            device,
            splits[2],
            args.batch_size,
        )
        for domain_name, splits in domain_splits.items()
    }

    save_json({"history": history}, output_dir / "training_history.json")
    write_metrics(
        {
            "experiment": experiment_name,
            "datasets": dataset_names,
            "domain_metrics": domain_metrics,
        },
        history[-1]["train_loss"],
        history[-1]["valid_loss"],
        test_loss,
        generation_metrics,
        output_dir / "metrics.json",
    )
    save_json(generation_metrics, output_dir / "generation_metrics.json")
    save_json(domain_metrics, output_dir / "domain_metrics.json")

    write_sample_conversations(
        model,
        vocabulary,
        device,
        test_examples,
        output_dir / "sample_conversations.txt",
    )

    reference_diversity = {
        domain_name: {
            "distinct_1": round(
                ngram_diversity(
                    (tokenize(example.target_text) for example in splits[2]), 1
                ),
                4,
            ),
            "distinct_2": round(
                ngram_diversity(
                    (tokenize(example.target_text) for example in splits[2]), 2
                ),
                4,
            ),
        }
        for domain_name, splits in domain_splits.items()
    }
    reference_diversity["combined"] = {
        "distinct_1": round(
            ngram_diversity(
                (tokenize(example.target_text) for example in test_examples), 1
            ),
            4,
        ),
        "distinct_2": round(
            ngram_diversity(
                (tokenize(example.target_text) for example in test_examples), 2
            ),
            4,
        ),
    }
    save_json(reference_diversity, output_dir / "reference_diversity.json")

    return {
        "experiment": experiment_name,
        "datasets": dataset_names,
        "train_loss": round(history[-1]["train_loss"], 4),
        "valid_loss": round(history[-1]["valid_loss"], 4),
        "test_loss": round(test_loss, 4),
        "generation_metrics": generation_metrics,
        "domain_metrics": domain_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise 3 pipeline: multi-dataset training and domain-effect analysis."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS.keys()),
        default=["cornell_only", "cornell_plus_persona"],
        help="Which Ex3 experiments to run.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Optional Cornell cap for quicker smoke tests.",
    )
    parser.add_argument(
        "--max-persona-dialogues",
        type=int,
        default=None,
        help="Optional PersonaChat cap for quicker smoke tests.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=config.EX3_OUTPUT_DIR)
    parser.add_argument(
        "--use-glove",
        action="store_true",
        default=config.USE_GLOVE,
        help="Initialize embeddings with pretrained GloVe vectors when available.",
    )
    parser.add_argument(
        "--no-glove",
        action="store_false",
        dest="use_glove",
        help="Disable GloVe initialization and train embeddings from scratch.",
    )
    parser.add_argument("--glove-path", type=Path, default=config.GLOVE_FILE)
    parser.add_argument("--glove-dim", type=int, default=config.GLOVE_DIM)
    parser.add_argument(
        "--freeze-glove",
        action="store_true",
        default=config.FREEZE_GLOVE,
        help="Freeze embedding weights after GloVe initialization.",
    )
    parser.add_argument("--layers", type=int, default=config.MODEL_NUM_LAYERS)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.EX3_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.EX3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.EX3_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for experiment_name in args.experiments:
        results.append(
            run_experiment(
                experiment_name,
                EXPERIMENTS[experiment_name],
                args,
                device,
            )
        )

    save_json({"results": results}, args.output_dir / "comparison.json")


if __name__ == "__main__":
    main()

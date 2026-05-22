import argparse
from pathlib import Path
import sys

import torch

from src import config
from src.feedback_ex6 import FeedbackStore, apply_feedback
from src.text import tokenize, detokenize
from src.vocab import Vocabulary
from src.model import Seq2SeqChatbot


def _find_checkpoint_path(requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            config.EX6_CHECKPOINT_DIR / "seq2seq_attention.pt",
            config.EX4_CHECKPOINT_DIR / "seq2seq_attention.pt",
            config.EX3_CHECKPOINT_DIR / "cornell_plus_persona" / "seq2seq_attention.pt",
            config.EX2_CHECKPOINT_DIR / "seq2seq_attention.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No chatbot checkpoint found for inference.")


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[Seq2SeqChatbot, Vocabulary]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocabulary = Vocabulary(checkpoint["vocab"])
    model = Seq2SeqChatbot(
        vocab_size=len(vocabulary.token_to_id),
        embed_dim=checkpoint.get("embed_dim", config.EMBED_DIM),
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        num_layers=checkpoint.get("num_layers", config.MODEL_NUM_LAYERS),
        bidirectional_encoder=config.USE_BIDIRECTIONAL_ENCODER,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary


def tensorize_example_local(example_text: str, vocabulary, device: torch.device):
    source_tokens = tokenize(example_text)[: config.MAX_SOURCE_TOKENS]
    source_ids = vocabulary.encode(source_tokens, add_eos=True)
    return (
        torch.tensor([source_ids], dtype=torch.long, device=device),
        torch.tensor([len(source_ids)], dtype=torch.long, device=device),
    )


def decode_ids_local(vocabulary, token_ids: list[int]) -> str:
    return detokenize(vocabulary.decode(token_ids, stop_at_eos=True))


def infer_once(
    model,
    vocabulary,
    device,
    feedback_store: FeedbackStore,
    query: str,
    method: str = "greedy",
) -> str:
    source_ids, source_length = tensorize_example_local(query, vocabulary, device)
    if method == "greedy":
        gen = model.greedy_decode(
            source_ids,
            source_length,
            max_length=config.MAX_TARGET_TOKENS,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
        )
        text = decode_ids_local(vocabulary, gen.token_ids)
    elif method == "beam":
        gen = model.beam_search(
            source_ids,
            source_length,
            max_length=config.MAX_TARGET_TOKENS,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
        )
        text = decode_ids_local(vocabulary, gen.token_ids)
    else:
        gen = model.sample_decode(
            source_ids,
            source_length,
            max_length=config.MAX_TARGET_TOKENS,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
            top_k=30,
            temperature=0.8,
        )
        text = decode_ids_local(vocabulary, gen.token_ids)

    return apply_feedback(query, feedback_store, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive inference with feedback store."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--query", type=str, default=None, help="Single query to run and exit"
    )
    parser.add_argument(
        "--method", type=str, default="greedy", choices=("greedy", "beam", "sample")
    )
    parser.add_argument("--feedback-path", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = _find_checkpoint_path(args.checkpoint)
    model, vocabulary = _load_model(checkpoint_path, device)

    feedback_store = (
        FeedbackStore(path=args.feedback_path)
        if args.feedback_path
        else FeedbackStore()
    )

    if args.query:
        out = infer_once(
            model, vocabulary, device, feedback_store, args.query, method=args.method
        )
        print(out)
        return

    # Interactive loop
    print(
        "Interactive inference with feedback. Type 'quit' to exit. To add a correction, use: correct: <your corrected response>"
    )
    while True:
        try:
            user = input("USER: ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            break
        if user.lower().startswith("correct:"):
            # Add correction for last query (assume last_query present)
            try:
                parts = user.split(":", 1)
                corrected = parts[1].strip()
                if "last_query" in globals() and globals()["last_query"]:
                    feedback_store.add_correction(globals()["last_query"], corrected)
                    print("Correction saved.")
                else:
                    print(
                        "No previous query to attach correction to. Use: correct: <correction> after asking a question."
                    )
            except Exception as e:
                print("Failed to save correction:", e)
            continue
        # Generate
        globals()["last_query"] = user
        out = infer_once(
            model, vocabulary, device, feedback_store, user, method=args.method
        )
        print("BOT:", out)


if __name__ == "__main__":
    main()

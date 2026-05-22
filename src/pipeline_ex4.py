from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src import config
from src.cornell import save_json
from src.dataset import DialogueExample
from src.memory_ex4 import (
    MemoryState,
    answer_from_memory,
    build_memory_prompt,
    is_memory_question,
)
from src.model import Seq2SeqChatbot
from src.pipeline import (
    decode_ids,
    maybe_load_glove_matrix,
    set_seed,
    tensorize_example,
)
from src.text import tokenize
from src.vocab import Vocabulary


@dataclass
class MemoryTestCase:
    name: str
    turns: list[str]
    expected_fact: str
    expected_answer: str


DEFAULT_TEST_CASES: list[MemoryTestCase] = [
    MemoryTestCase(
        name="name_recall",
        turns=["My name is Ana.", "What is my name?"],
        expected_fact="name",
        expected_answer="Your name is Ana.",
    ),
    MemoryTestCase(
        name="location_recall",
        turns=["I live in Lyon.", "Where do I live?"],
        expected_fact="location",
        expected_answer="You live in Lyon.",
    ),
    MemoryTestCase(
        name="preference_recall",
        turns=["I love pizza.", "What do I like?"],
        expected_fact="preference",
        expected_answer="You like pizza.",
    ),
    MemoryTestCase(
        name="statement_recall",
        turns=["I am learning piano.", "What did I tell you?"],
        expected_fact="identity",
        expected_answer="You said: I am learning piano.",
    ),
    MemoryTestCase(
        name="combined_recall",
        turns=[
            "My name is Sam. I live in Toronto. I like hiking.",
            "Do you remember me?",
        ],
        expected_fact="name",
        expected_answer="Yes, I remember: location=Toronto; name=Sam; preference=hiking.",
    ),
]


def _find_checkpoint_path(requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            config.EX4_CHECKPOINT_DIR / "seq2seq_attention.pt",
            config.EX3_CHECKPOINT_DIR / "cornell_plus_persona" / "seq2seq_attention.pt",
            config.EX2_CHECKPOINT_DIR / "seq2seq_attention.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No chatbot checkpoint found for Exercise 4.")


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[Seq2SeqChatbot, Vocabulary]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocabulary = Vocabulary(checkpoint["vocab"])
    model = Seq2SeqChatbot(
        vocab_size=len(vocabulary.token_to_id),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        num_layers=checkpoint["num_layers"],
        bidirectional_encoder=config.USE_BIDIRECTIONAL_ENCODER,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary


def _decode_model_response(
    model: Seq2SeqChatbot,
    vocabulary: Vocabulary,
    device: torch.device,
    user_text: str,
    memory: MemoryState | None = None,
) -> str:
    prompt = build_memory_prompt(user_text, memory) if memory is not None else user_text
    source_ids, source_length = tensorize_example(prompt, vocabulary, device)
    generation = model.greedy_decode(
        source_ids,
        source_length,
        max_length=config.MAX_TARGET_TOKENS,
        bos_id=vocabulary.bos_id,
        eos_id=vocabulary.eos_id,
    )
    return decode_ids(vocabulary, generation.token_ids)


def run_memory_turns(
    model: Seq2SeqChatbot,
    vocabulary: Vocabulary,
    device: torch.device,
    turns: list[str],
) -> dict[str, object]:
    memory = MemoryState()
    baseline_responses: list[str] = []
    memory_responses: list[str] = []
    memory_states: list[dict[str, str]] = []

    for turn in turns:
        if memory is not None:
            if not is_memory_question(turn):
                memory.update_from_user(turn)
            memory_answer = answer_from_memory(turn, memory)
            if memory_answer is not None:
                memory_responses.append(memory_answer)
            else:
                memory_responses.append(
                    _decode_model_response(model, vocabulary, device, turn, memory)
                )
            memory_states.append(dict(memory.facts))
        baseline_responses.append(
            _decode_model_response(model, vocabulary, device, turn)
        )

    return {
        "turns": turns,
        "baseline_responses": baseline_responses,
        "memory_responses": memory_responses,
        "memory_states": memory_states,
    }


def run_test_cases(
    model: Seq2SeqChatbot,
    vocabulary: Vocabulary,
    device: torch.device,
    test_cases: list[MemoryTestCase],
) -> dict[str, object]:
    results = []
    passed = 0

    for test_case in test_cases:
        turn_result = run_memory_turns(model, vocabulary, device, test_case.turns)
        memory_response = turn_result["memory_responses"][-1]
        is_pass = memory_response == test_case.expected_answer
        passed += int(is_pass)
        results.append(
            {
                "name": test_case.name,
                "turns": test_case.turns,
                "expected_fact": test_case.expected_fact,
                "expected_answer": test_case.expected_answer,
                "memory_response": memory_response,
                "baseline_response": turn_result["baseline_responses"][-1],
                "passed": is_pass,
                "memory_states": turn_result["memory_states"],
            }
        )

    return {
        "num_cases": len(test_cases),
        "num_passed": passed,
        "accuracy": round(passed / max(1, len(test_cases)), 4),
        "results": results,
    }


def write_report(results: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Exercise 4 memory comparison\n")
        handle.write("=" * 32 + "\n\n")
        handle.write(
            f"Accuracy: {results['accuracy']} ({results['num_passed']}/{results['num_cases']})\n\n"
        )
        for case_result in results["results"]:
            handle.write(f"Case: {case_result['name']}\n")
            handle.write(
                f"Turns: {json.dumps(case_result['turns'], ensure_ascii=False)}\n"
            )
            handle.write(f"Baseline: {case_result['baseline_response']}\n")
            handle.write(f"Memory: {case_result['memory_response']}\n")
            handle.write(f"Passed: {case_result['passed']}\n")
            handle.write("-" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise 4 pipeline: simple rule-based conversational memory."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=config.EX4_OUTPUT_DIR)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.EX4_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.EX4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.EX4_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _find_checkpoint_path(args.checkpoint)
    model, vocabulary = _load_model(checkpoint_path, device)

    results = run_test_cases(model, vocabulary, device, DEFAULT_TEST_CASES)
    save_json(
        {"checkpoint": str(checkpoint_path), **results},
        args.output_dir / "memory_test_results.json",
    )
    write_report(results, args.output_dir / "memory_test_report.txt")


if __name__ == "__main__":
    main()

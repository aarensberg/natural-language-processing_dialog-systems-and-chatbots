from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src import config

EX1_METRICS = config.EX1_OUTPUT_DIR / "metrics.json"
EX3_CORNELL_ONLY = config.EX3_OUTPUT_DIR / "cornell_only" / "metrics.json"
EX3_CORNELL_PLUS = config.EX3_OUTPUT_DIR / "cornell_plus_persona" / "metrics.json"
EX4_RESULTS = config.EX4_OUTPUT_DIR / "memory_test_results.json"
EX5_RESULTS = config.EX5_OUTPUT_DIR / "personality_test_results.json"
EX6_RESULTS = config.EX6_OUTPUT_DIR / "large_eval_results.json"
EX6_GENERALIZATION = config.EX6_OUTPUT_DIR / "generalization_results.json"
EX1_SAMPLES = config.EX1_OUTPUT_DIR / "sample_conversations.txt"
EX5_SAMPLES = config.EX5_OUTPUT_DIR / "sample_conversations.txt"
EX6_FAILURES = config.EX6_OUTPUT_DIR / "large_eval_failure_cases.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def extract_sample_block(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    index = text.index(marker)
    tail = text[index:]
    lines = tail.splitlines()
    return "\n".join(lines[:16])


def build_variant_comparison() -> list[dict]:
    ex1 = load_json(EX1_METRICS)
    ex3_cornell_only = load_json(EX3_CORNELL_ONLY)
    ex3_cornell_plus = load_json(EX3_CORNELL_PLUS)
    ex4 = load_json(EX4_RESULTS)
    ex5 = load_json(EX5_RESULTS)
    ex6 = load_json(EX6_RESULTS)
    ex6_generalization = load_json(EX6_GENERALIZATION)

    variants = [
        {
            "name": "baseline_ex1",
            "type": "neural baseline",
            "source": "Cornell only",
            "test_loss": ex1["test_loss"],
            "test_perplexity": ex1["test_perplexity"],
            "primary_metric": "test_loss",
            "qualitative_note": "word-level GRU baseline; strongest weakness is generic responses and weak grounding",
        },
        {
            "name": "improved_model_ex3",
            "type": "attention + extra data",
            "source": "Cornell + PersonaChat",
            "test_loss": ex3_cornell_plus["test_loss"],
            "test_perplexity": ex3_cornell_plus["test_perplexity"],
            "primary_metric": "test_loss",
            "generation_metrics": ex3_cornell_plus["generation_metrics"],
            "qualitative_note": "better loss than baseline; top-k decoding increases diversity most strongly",
        },
        {
            "name": "memory_augmented",
            "type": "rule-based memory",
            "source": "same seq2seq core + memory",
            "accuracy": ex4["accuracy"],
            "num_cases": ex4["num_cases"],
            "baseline_accuracy": safe_ratio(
                sum(
                    1
                    for case in ex4["results"]
                    if case["baseline_response"] == case["expected_answer"]
                ),
                ex4["num_cases"],
            ),
            "qualitative_note": "perfect on the five factual recall tests; baseline failed every case",
        },
        {
            "name": "persona_augmented",
            "type": "persona profile + rules",
            "source": "same seq2seq core + persona",
            "accuracy": ex5["accuracy"],
            "num_cases": ex5["num_cases"],
            "baseline_accuracy": safe_ratio(
                sum(
                    1
                    for case in ex5["results"]
                    if case["baseline_responses"][0] == case["expected_phrase"]
                ),
                ex5["num_cases"],
            ),
            "qualitative_note": "stable persona is fully enforced by rules plus prompt conditioning",
        },
        {
            "name": "feedback_augmented",
            "type": "persistent feedback store",
            "source": "same seq2seq core + feedback",
            "positive_coverage": ex6["stats"]["positive_coverage"],
            "false_positive_rate": ex6["stats"]["false_positive_rate"],
            "overall_accuracy": ex6["stats"]["overall_accuracy"],
            "avg_similarity": ex6["stats"]["avg_similarity"],
            "generalization_coverage": ex6_generalization["stats"]["coverage"],
            "qualitative_note": "strong correction coverage, but semantic over-match can trigger false positives",
        },
    ]
    return variants


def build_ablation_study() -> list[dict]:
    ex3_cornell_only = load_json(EX3_CORNELL_ONLY)
    ex3_cornell_plus = load_json(EX3_CORNELL_PLUS)
    ex4 = load_json(EX4_RESULTS)
    ex5 = load_json(EX5_RESULTS)
    ex6 = load_json(EX6_RESULTS)

    persona_baseline = safe_ratio(
        sum(
            1
            for case in ex5["results"]
            if case["baseline_responses"][0] == case["expected_phrase"]
        ),
        ex5["num_cases"],
    )
    memory_baseline = safe_ratio(
        sum(
            1
            for case in ex4["results"]
            if case["baseline_response"] == case["expected_answer"]
        ),
        ex4["num_cases"],
    )
    feedback_before_exact = safe_ratio(
        sum(
            1
            for record in load_json(EX6_RESULTS)["records"]
            if record["expected"] is not None
            and record["baseline_generated"] == record["expected"]
        ),
        load_json(EX6_RESULTS)["stats"]["positive_total"],
    )

    ablations = [
        {
            "name": "dataset_ablation",
            "full_setting": "Cornell + PersonaChat",
            "ablated_setting": "Cornell only",
            "full_test_loss": ex3_cornell_plus["test_loss"],
            "ablated_test_loss": ex3_cornell_only["test_loss"],
            "full_test_perplexity": ex3_cornell_plus["test_perplexity"],
            "ablated_test_perplexity": ex3_cornell_only["test_perplexity"],
            "delta_test_loss": round(
                ex3_cornell_plus["test_loss"] - ex3_cornell_only["test_loss"], 4
            ),
            "delta_perplexity": round(
                ex3_cornell_plus["test_perplexity"]
                - ex3_cornell_only["test_perplexity"],
                4,
            ),
        },
        {
            "name": "memory_ablation",
            "full_setting": "memory enabled",
            "ablated_setting": "memory removed",
            "full_accuracy": ex4["accuracy"],
            "ablated_accuracy": memory_baseline,
            "delta_accuracy": round(ex4["accuracy"] - memory_baseline, 4),
        },
        {
            "name": "persona_ablation",
            "full_setting": "persona enabled",
            "ablated_setting": "persona removed",
            "full_accuracy": ex5["accuracy"],
            "ablated_accuracy": persona_baseline,
            "delta_accuracy": round(ex5["accuracy"] - persona_baseline, 4),
        },
        {
            "name": "feedback_ablation",
            "full_setting": "semantic feedback enabled",
            "ablated_setting": "feedback absent / before corrections",
            "full_positive_coverage": ex6["stats"]["positive_coverage"],
            "full_false_positive_rate": ex6["stats"]["false_positive_rate"],
            "before_exact_match_rate": feedback_before_exact,
            "delta_vs_before": round(
                ex6["stats"]["positive_coverage"] - feedback_before_exact, 4
            ),
        },
    ]
    return ablations


def build_failure_analysis() -> list[dict]:
    failures = load_json(EX6_FAILURES)
    selected = failures[:5]
    analysis = []
    for failure in selected:
        analysis.append(
            {
                "id": failure["id"],
                "category": failure["category"],
                "error_type": failure["error_type"],
                "query": failure["query"],
                "expected": failure["expected"],
                "before": failure["before"],
                "after": failure["after"],
                "strategy": failure.get("strategy"),
                "similarity": failure.get("similarity"),
                "diagnosis": (
                    "semantic over-match"
                    if failure["error_type"] == "false_positive"
                    else "missed paraphrase / underspecified self-description"
                ),
            }
        )
    return analysis


def build_qualitative_examples() -> dict[str, str]:
    samples_ex1 = load_text(EX1_SAMPLES)
    samples_ex5 = load_text(EX5_SAMPLES)
    large_eval = load_json(EX6_RESULTS)
    large_eval_text = load_text(config.EX6_OUTPUT_DIR / "large_eval_before_after.txt")

    return {
        "baseline_success": extract_sample_block(samples_ex1, "USER:")
        or samples_ex1[:800],
        "persona_success": extract_sample_block(samples_ex5, "CONVERSATION 1")
        or samples_ex5[:800],
        "feedback_before_after": extract_sample_block(
            large_eval_text, "CASE: food_exact"
        )
        or large_eval_text[:800],
        "feedback_failure": json.dumps(
            large_eval["failure_cases"][:2], indent=2, ensure_ascii=False
        ),
    }


def build_summary() -> dict:
    variants = build_variant_comparison()
    ablations = build_ablation_study()
    failures = build_failure_analysis()
    examples = build_qualitative_examples()

    summary = {
        "variants": variants,
        "ablations": ablations,
        "failure_cases": failures,
        "examples": examples,
    }
    return summary


def write_report(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ex7_report.txt"
    json_path = output_dir / "ex7_summary.json"
    samples_path = output_dir / "ex7_examples.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    lines: list[str] = []
    lines.append("Exercise 7 evaluation, ablation, and error analysis")
    lines.append("=" * 55)
    lines.append("")
    lines.append("Variant comparison")
    lines.append("------------------")
    for variant in summary["variants"]:
        lines.append(f"Name: {variant['name']}")
        lines.append(f"Type: {variant['type']}")
        lines.append(f"Source: {variant['source']}")
        if "test_loss" in variant:
            lines.append(f"Test loss: {variant['test_loss']:.4f}")
            lines.append(f"Test perplexity: {variant['test_perplexity']:.4f}")
            if "generation_metrics" in variant:
                lines.append(
                    "Greedy BLEU / distinct-2: "
                    f"{variant['generation_metrics']['greedy']['bleu']:.4f} / {variant['generation_metrics']['greedy']['distinct_2']:.4f}"
                )
                lines.append(
                    "Top-k BLEU / distinct-2: "
                    f"{variant['generation_metrics']['topk_temp']['bleu']:.4f} / {variant['generation_metrics']['topk_temp']['distinct_2']:.4f}"
                )
        if "accuracy" in variant:
            lines.append(f"Accuracy: {variant['accuracy']:.4f}")
            lines.append(f"Baseline accuracy: {variant['baseline_accuracy']:.4f}")
        if "positive_coverage" in variant:
            lines.append(f"Positive coverage: {variant['positive_coverage']:.4f}")
            lines.append(f"False positive rate: {variant['false_positive_rate']:.4f}")
            lines.append(f"Overall accuracy: {variant['overall_accuracy']:.4f}")
            lines.append(f"Average similarity: {variant['avg_similarity']:.4f}")
            lines.append(
                f"Generalization coverage: {variant['generalization_coverage']:.4f}"
            )
        lines.append(f"Note: {variant['qualitative_note']}")
        lines.append("")

    lines.append("Ablation study")
    lines.append("--------------")
    for ablation in summary["ablations"]:
        lines.append(f"Name: {ablation['name']}")
        lines.append(f"Full setting: {ablation['full_setting']}")
        lines.append(f"Ablated setting: {ablation['ablated_setting']}")
        for key, value in ablation.items():
            if key in {"name", "full_setting", "ablated_setting"}:
                continue
            lines.append(f"{key}: {value}")
        lines.append("")

    lines.append("Failure analysis")
    lines.append("---------------")
    for failure in summary["failure_cases"]:
        lines.append(
            f"[{failure['error_type']}] {failure['id']} ({failure['category']})"
        )
        lines.append(f"Query: {failure['query']}")
        lines.append(f"Expected: {failure['expected']}")
        lines.append(f"Before: {failure['before']}")
        lines.append(f"After: {failure['after']}")
        lines.append(
            f"Strategy: {failure['strategy']} | Similarity: {failure['similarity']}"
        )
        lines.append(f"Diagnosis: {failure['diagnosis']}")
        lines.append("")

    lines.append("Qualitative examples")
    lines.append("--------------------")
    for name, block in summary["examples"].items():
        lines.append(f"Example block: {name}")
        lines.append(block)
        lines.append("")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    with samples_path.open("w", encoding="utf-8") as handle:
        for name, block in summary["examples"].items():
            handle.write(f"[{name}]\n")
            handle.write(block)
            handle.write("\n" + "=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise 7 evaluation protocol")
    parser.add_argument("--output-dir", type=Path, default=config.EX7_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_summary()
    write_report(summary, args.output_dir)
    print(f"Wrote Exercise 7 summary to {args.output_dir}")


if __name__ == "__main__":
    main()

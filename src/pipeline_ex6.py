import json
from pathlib import Path
from typing import List

from .config import EX6_OUTPUT_DIR
from .feedback_ex6 import FeedbackStore, apply_feedback

DEFAULT_FEEDBACK_CASES = [
    {
        "user_query": "What is your favourite food?",
        "baseline_response": "I do not know.",
        "correction": "I like pizza.",
    },
    {
        "user_query": "What city do you live in?",
        "baseline_response": "I don't have one.",
        "correction": "I live in Lyon.",
    },
    {
        "user_query": "What's your job?",
        "baseline_response": "I am not sure.",
        "correction": "I am a calm chatbot research assistant.",
    },
    {
        "user_query": "What's your favourite drink?",
        "baseline_response": "I like many things.",
        "correction": "I prefer tea.",
    },
    {
        "user_query": "What is your name?",
        "baseline_response": "I don't remember.",
        "correction": "My name is Ari.",
    },
]


def run_feedback_smoketest(cases: List[dict] = None) -> None:
    cases = cases or DEFAULT_FEEDBACK_CASES
    out_dir = Path(EX6_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    feedback_store = FeedbackStore(path=out_dir / "feedback_store.json")

    before = []
    for c in cases:
        before.append(
            {
                "user_query": c["user_query"],
                "baseline_response": c["baseline_response"],
            }
        )

    with open(out_dir / "feedback_before.json", "w", encoding="utf-8") as f:
        json.dump(before, f, indent=2, ensure_ascii=False)

    for c in cases:
        feedback_store.add_correction(c["user_query"], c["correction"])

    after = []
    num_changed = 0
    for c in cases:
        corrected = apply_feedback(
            c["user_query"], feedback_store, c["baseline_response"]
        )
        after.append(
            {
                "user_query": c["user_query"],
                "response_after": corrected,
                "expected": c["correction"],
                "matched": corrected == c["correction"],
            }
        )
        if corrected == c["correction"]:
            num_changed += 1

    with open(out_dir / "feedback_after.json", "w", encoding="utf-8") as f:
        json.dump(after, f, indent=2, ensure_ascii=False)

    results = {
        "num_cases": len(cases),
        "num_corrected": num_changed,
        "accuracy": num_changed / len(cases) if cases else 0.0,
        "feedback_store": str(out_dir / "feedback_store.json"),
    }

    with open(out_dir / "feedback_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("Exercise 6: Feedback smoke test")
    lines.append("================================\n")
    lines.append(
        f"Corrected: {num_changed}/{len(cases)} | accuracy={results['accuracy']:.2f}\n"
    )
    for a in after:
        lines.append("Case:")
        lines.append(f"User: {a['user_query']}")
        lines.append(f"After: {a['response_after']}")
        lines.append(f"Expected: {a['expected']}")
        lines.append(f"Matched: {a['matched']}\n")

    with open(out_dir / "feedback_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(out_dir / "sample_before_after.txt", "w", encoding="utf-8") as f:
        for b, a in zip(before, after):
            f.write("USER: " + b["user_query"] + "\n")
            f.write("BEFORE: " + b["baseline_response"] + "\n")
            f.write("AFTER: " + a["response_after"] + "\n")
            f.write("-" * 72 + "\n")


def extended_cases() -> List[dict]:
    return [
        {
            "user_query": "What's your favorite food?",
            "baseline_response": "I do not know.",
            "correction": "I like pizza.",
        },
        {
            "user_query": "Which city are you living in?",
            "baseline_response": "I don't have one.",
            "correction": "I live in Lyon.",
        },
        {
            "user_query": "Tell me your job.",
            "baseline_response": "I am not sure.",
            "correction": "I am a calm chatbot research assistant.",
        },
        {
            "user_query": "Favorite drink?",
            "baseline_response": "I like many things.",
            "correction": "I prefer tea.",
        },
        {
            "user_query": "Who are you?",
            "baseline_response": "I don't remember.",
            "correction": "My name is Ari.",
        },
        {
            "user_query": "Could you tell me what food you like most?",
            "baseline_response": "I do not know.",
            "correction": "I like pizza.",
        },
        {
            "user_query": "In which city are you based?",
            "baseline_response": "I don't have one.",
            "correction": "I live in Lyon.",
        },
        {
            "user_query": "What do you do for work?",
            "baseline_response": "I am not sure.",
            "correction": "I am a calm chatbot research assistant.",
        },
        {
            "user_query": "What drink do you prefer?",
            "baseline_response": "I like many things.",
            "correction": "I prefer tea.",
        },
        {
            "user_query": "Tell me your name.",
            "baseline_response": "I don't remember.",
            "correction": "My name is Ari.",
        },
    ]


def run_generalization_test(
    canonical_cases: List[dict], paraphrase_groups: dict, out_dir: Path
) -> dict:
    feedback_store = FeedbackStore(path=out_dir / "feedback_store.json")

    results = []
    total_paraphrases = 0
    exact_matches = 0
    semantic_matches = 0
    unmatched = 0
    similarity_sum = 0.0

    for canonical in canonical_cases:
        key = canonical["user_query"]
        expected = canonical["correction"]
        paraphrases = paraphrase_groups.get(key, [])
        for p in paraphrases:
            total_paraphrases += 1
            result = feedback_store.resolve(p, min_similarity=0.35)
            corr = result.get("correction")
            strategy = result.get("strategy")
            similarity = float(result.get("similarity") or 0.0)
            similarity_sum += similarity
            matched = corr == expected
            direct_hit = strategy == "exact" and matched
            semantic_hit = strategy == "semantic" and matched
            if strategy == "exact":
                exact_matches += 1
            elif strategy == "semantic":
                semantic_matches += 1
            else:
                unmatched += 1
            results.append(
                {
                    "canonical": key,
                    "paraphrase": p,
                    "correction_found": corr,
                    "expected": expected,
                    "strategy": strategy,
                    "similarity": round(similarity, 4),
                    "matched": matched,
                    "direct_hit": direct_hit,
                    "semantic_hit": semantic_hit,
                }
            )

    matched_total = exact_matches + semantic_matches
    coverage = matched_total / total_paraphrases if total_paraphrases else 0.0
    stats = {
        "total_paraphrases": total_paraphrases,
        "matched": matched_total,
        "exact_matches": exact_matches,
        "semantic_matches": semantic_matches,
        "unmatched": unmatched,
        "coverage": coverage,
        "avg_similarity": (
            similarity_sum / total_paraphrases if total_paraphrases else 0.0
        ),
    }

    with open(out_dir / "generalization_results.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "details": results}, f, indent=2, ensure_ascii=False)

    with open(out_dir / "generalization_report.txt", "w", encoding="utf-8") as f:
        f.write("Generalization test for feedback paraphrases\n")
        f.write("=" * 48 + "\n\n")
        f.write(
            f"Matched paraphrases: {matched_total}/{total_paraphrases} | coverage={coverage:.3f}\n"
        )
        f.write(
            f"Exact matches: {exact_matches} | Semantic matches: {semantic_matches} | Unmatched: {unmatched}\n"
        )
        f.write(f"Average similarity: {stats['avg_similarity']:.3f}\n\n")
        for r in results:
            f.write(f"Canonical: {r['canonical']}\n")
            f.write(f"Paraphrase: {r['paraphrase']}\n")
            f.write(
                f"Expected: {r['expected']} | Found: {r['correction_found']} | strategy: {r['strategy']} | similarity: {r['similarity']}\n"
            )
            f.write("-" * 48 + "\n")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exercise 6 feedback smoke test")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended paraphrase/generalization cases",
    )
    args = parser.parse_args()
    if args.extended:
        run_feedback_smoketest()
        run_feedback_smoketest(extended_cases())
        paraphrase_map = {
            DEFAULT_FEEDBACK_CASES[0]["user_query"]: [
                "What's your favorite food?",
                "What's your favourite food?",
                "Could you tell me what food you like most?",
            ],
            DEFAULT_FEEDBACK_CASES[1]["user_query"]: [
                "Which city are you living in?",
                "What city do you live in?",
                "In which city are you based?",
            ],
            DEFAULT_FEEDBACK_CASES[2]["user_query"]: [
                "Tell me your job.",
                "What's your job?",
                "What do you do for work?",
            ],
            DEFAULT_FEEDBACK_CASES[3]["user_query"]: [
                "Favorite drink?",
                "What's your favourite drink?",
                "What drink do you prefer?",
            ],
            DEFAULT_FEEDBACK_CASES[4]["user_query"]: [
                "Who are you?",
                "What is your name?",
                "Tell me your name.",
            ],
        }
        stats = run_generalization_test(
            DEFAULT_FEEDBACK_CASES, paraphrase_map, Path(EX6_OUTPUT_DIR)
        )
        print(f"Generalization coverage: {stats['coverage']:.3f}")
    else:
        run_feedback_smoketest()

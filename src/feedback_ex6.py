import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional

from .config import EX6_OUTPUT_DIR


class FeedbackStore:
    """Simple persistent store for user corrections.

    Maps normalized user queries to corrected responses and saves to JSON.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = (
            Path(path) if path is not None else EX6_OUTPUT_DIR / "feedback_store.json"
        )
        self._data: Dict[str, str] = {}
        self._semantic_cache = None
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def add_correction(self, user_query: str, corrected_response: str) -> None:
        key = self.normalize(user_query)
        self._data[key] = corrected_response
        self._semantic_cache = None
        self._save()

    def get_correction(self, user_query: str) -> Optional[str]:
        return self.resolve(user_query).get("correction")

    def _build_semantic_cache(self):
        # Cache the normalized queries so semantic lookup can compare against them quickly.
        if self._semantic_cache is None:
            keys = list(self._data.keys())
            self._semantic_cache = keys if keys else None
        return self._semantic_cache

    def _token_overlap_score(self, left: str, right: str) -> float:
        left_tokens = left.split()
        right_tokens = right.split()
        if not left_tokens or not right_tokens:
            return 0.0
        left_counts = Counter(left_tokens)
        right_counts = Counter(right_tokens)
        overlap = sum((left_counts & right_counts).values())
        denominator = max(len(left_tokens), len(right_tokens))
        return overlap / denominator if denominator else 0.0

    def _semantic_similarity(self, left: str, right: str) -> float:
        char_ratio = SequenceMatcher(None, left, right).ratio()
        token_overlap = self._token_overlap_score(left, right)
        return 0.7 * char_ratio + 0.3 * token_overlap

    def resolve(
        self, user_query: str, *, min_similarity: float = 0.55
    ) -> dict[str, object]:
        normalized = self.normalize(user_query)
        exact = self._data.get(normalized)
        if exact is not None:
            return {
                "correction": exact,
                "matched_query": normalized,
                "similarity": 1.0,
                "strategy": "exact",
            }

        semantic_cache = self._build_semantic_cache()
        if semantic_cache is None:
            return {
                "correction": None,
                "matched_query": None,
                "similarity": 0.0,
                "strategy": None,
            }

        keys = semantic_cache
        similarities = [self._semantic_similarity(normalized, key) for key in keys]
        if not similarities:
            return {
                "correction": None,
                "matched_query": None,
                "similarity": 0.0,
                "strategy": None,
            }

        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        best_similarity = float(similarities[best_index])
        if best_similarity < min_similarity:
            return {
                "correction": None,
                "matched_query": None,
                "similarity": best_similarity,
                "strategy": None,
            }

        matched_query = keys[best_index]
        return {
            "correction": self._data.get(matched_query),
            "matched_query": matched_query,
            "similarity": best_similarity,
            "strategy": "semantic",
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def export(self) -> Dict[str, str]:
        return dict(self._data)


def apply_feedback(
    user_query: str,
    feedback_store: FeedbackStore,
    fallback_response: str,
    *,
    min_similarity: float = 0.55,
) -> str:
    """Return corrected response if available, otherwise fallback_response."""
    result = feedback_store.resolve(user_query, min_similarity=min_similarity)
    corr = result.get("correction")
    return corr if corr is not None else fallback_response

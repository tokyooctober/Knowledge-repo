"""The judge-independent retrieval metrics are pure and app-venv-safe, so they get
a real unit test (the rest of eval/ is verified by running it — see eval/README.md)."""

from __future__ import annotations

from eval.metrics_simple import retrieval_metrics


def _row(retrieved: list[str], gold: list[str]) -> dict:
    return {"user_input": "q", "retrieved_urls": retrieved, "source_urls": gold}


def test_perfect_retrieval_scores_one() -> None:
    rows = [_row(["u1"], ["u1"]), _row(["u2", "u9"], ["u2"])]
    m = retrieval_metrics(rows)
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 1.0
    assert m["recall_at_k"] == 1.0


def test_gold_at_rank_two_halves_mrr() -> None:
    m = retrieval_metrics([_row(["wrong", "right"], ["right"])])
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 0.5
    assert m["precision_at_k"] == 0.5


def test_total_miss_scores_zero() -> None:
    m = retrieval_metrics([_row(["a", "b"], ["c"])])
    assert m == {"n": 1, "hit_rate": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0}


def test_rows_without_gold_are_skipped() -> None:
    rows = [_row(["a"], []), _row(["b"], ["b"])]
    m = retrieval_metrics(rows)
    assert m["n"] == 1
    assert m["hit_rate"] == 1.0


def test_no_scorable_rows_returns_note() -> None:
    m = retrieval_metrics([{"user_input": "q", "retrieved_urls": ["a"], "source_urls": []}])
    assert m["n"] == 0
    assert "note" in m

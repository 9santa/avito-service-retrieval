"""Оценивает RRF и union глобальных lexical baseline-ов."""

import pickle
from pathlib import Path

import pandas as pd

from src.metrics import build_labels, recall_at_k
from src.fusion import rrf_fusion, union_predictions


PRED_DIR = Path("artifacts/predictions")


def load_predictions(name):
    with open(PRED_DIR / name, "rb") as f:
        return pickle.load(f)


def main():
    val_part = pd.read_parquet("data/interim/val_part.parquet")

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    char = load_predictions("char_tfidf_val.pkl")
    word = load_predictions("word_tfidf_val.pkl")
    bm25 = load_predictions("bm25_val.pkl")

    experiments = {
        "word + bm25": [word, bm25],
        "char + word": [char, word],
        "char + bm25": [char, bm25],
        "char + word + bm25": [char, word, bm25],
    }

    for name, predictions in experiments.items():
        fused = rrf_fusion(
            predictions,
            k=50,
            rrf_k=60,
        )

        score = recall_at_k(
            labels,
            fused,
            k=50,
        )

        pool = union_predictions(predictions)

        # Здесь k ставим специально большим:
        # это не Recall@50, а recall всего union candidate pool
        pool_recall = recall_at_k(
            labels,
            pool,
            k=1000,
        )

        mean_pool_size = sum(len(x) for x in pool.values()) / len(pool)

        print(name)
        print(f"  RRF Recall@50: {score:.6f}")
        print(f"  Pool recall:    {pool_recall:.6f}")
        print(f"  Mean pool size: {mean_pool_size:.2f}")
        print()


if __name__ == "__main__":
    main()

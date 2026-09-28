"""Оценивает RRF и pool coverage двух location-aware lexical retriever-ов."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion import rrf_fusion, union_predictions
from src.metrics import build_labels, recall_at_k


VAL_PATH = Path("data/interim/val_part.parquet")
PRED_DIR = Path("artifacts/predictions")

WORD_PATH = PRED_DIR / "word_local_val.pkl"
CHAR_PATH = PRED_DIR / "char_local_val.pkl"


def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def mean_candidates(predictions):
    return np.mean([len(items) for items in predictions.values()])


def main():
    val_part = pd.read_parquet(VAL_PATH)

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    word = load_predictions(WORD_PATH)
    char = load_predictions(CHAR_PATH)

    print("Validation signatures:", len(labels))
    print()

    # ---------------------------------------------------------
    # Individual baselines
    # ---------------------------------------------------------

    for name, predictions in [
        ("word local", word),
        ("char local", char),
    ]:
        print(name)

        for k in [10, 20, 50]:
            score = recall_at_k(
                labels,
                predictions,
                k=k,
            )

            print(f"  Recall@{k}: {score:.6f}")

        print(f"  Mean candidates: {mean_candidates(predictions):.2f}")

        print()

    # ---------------------------------------------------------
    # RRF
    #
    # Оба retriever возвращают <=50 кандидатов,
    # поэтому их union содержит <=100 кандидатов.
    # ---------------------------------------------------------

    fused = rrf_fusion(
        [
            char,
            word,
        ],
        rrf_k=60,
        k=50,
    )

    print("RRF char + word")

    for k in [10, 20, 50]:
        score = recall_at_k(
            labels,
            fused,
            k=k,
        )

        print(f"  Recall@{k}: {score:.6f}")

    print(f"  Mean candidates: {mean_candidates(fused):.2f}")

    print()

    # ---------------------------------------------------------
    # Full union pool
    #
    # Это upper bound текущих двух candidate generators:
    # считаем recall по ВСЕМ уникальным кандидатам.
    # ---------------------------------------------------------

    pool = union_predictions(
        [
            char,
            word,
        ]
    )

    pool_recall = recall_at_k(
        labels,
        pool,
        k=1000,
    )

    pool_sizes = np.array([len(items) for items in pool.values()])

    print("Full local lexical pool")
    print(f"  Pool recall: {pool_recall:.6f}")
    print(f"  Mean pool size: {pool_sizes.mean():.2f}")
    print(f"  Median pool size: {np.median(pool_sizes):.2f}")
    print(f"  Max pool size: {pool_sizes.max()}")

    print()

    # ---------------------------------------------------------
    # Насколько retriever'ы пересекаются
    # ---------------------------------------------------------

    intersections = []
    unions = []

    for query_id in labels:
        word_set = set(word.get(query_id, []))
        char_set = set(char.get(query_id, []))

        intersections.append(len(word_set & char_set))

        unions.append(len(word_set | char_set))

    print("Candidate overlap")
    print(f"  Mean intersection: {np.mean(intersections):.2f}")
    print(f"  Mean union: {np.mean(unions):.2f}")


if __name__ == "__main__":
    main()

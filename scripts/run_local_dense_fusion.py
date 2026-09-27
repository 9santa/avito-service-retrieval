import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion import rrf_fusion, union_predictions
from src.metrics import build_labels, recall_at_k


VAL_PATH = Path("data/interim/val_part.parquet")
PRED_DIR = Path("artifacts/predictions")


FILES = {
    "char_local": PRED_DIR / "char_local_val.pkl",
    "word_local": PRED_DIR / "word_local_val.pkl",
    "dense_local": PRED_DIR / "dense_local_val.pkl",
    "dense_global": PRED_DIR / "dense_val.pkl",
}


def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def mean_candidates(predictions):
    return np.mean([len(items) for items in predictions.values()])


def evaluate_rrf(
    name,
    sources,
    labels,
    weights=None,
):
    fused = rrf_fusion(
        sources,
        rrf_k=60,
        k=50,
        weights=weights,
    )

    recall = recall_at_k(
        labels,
        fused,
        k=50,
    )

    print(f"{name:<45} RRF@50={recall:.6f}")

    return fused, recall


def evaluate_pool(
    name,
    sources,
    labels,
):
    pool = union_predictions(sources)

    recall = recall_at_k(
        labels,
        pool,
        k=1000,
    )

    sizes = np.array([len(items) for items in pool.values()])

    print(f"{name:<45} pool_recall={recall:.6f} | mean_pool={sizes.mean():.2f}")

    return pool, recall


def main():
    val_part = pd.read_parquet(VAL_PATH)

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    preds = {}

    for name, path in FILES.items():
        print(f"Loading {name}: {path}")
        preds[name] = load_predictions(path)

    char_local = preds["char_local"]
    word_local = preds["word_local"]
    dense_local = preds["dense_local"]
    dense_global = preds["dense_global"]

    print()
    print("=" * 90)
    print("STANDALONE")
    print("=" * 90)

    for name, pred in [
        ("char local", char_local),
        ("word local", word_local),
        ("dense local", dense_local),
        ("dense global", dense_global),
    ]:
        score = recall_at_k(
            labels,
            pred,
            k=50,
        )

        print(
            f"{name:<20} R@50={score:.6f} | mean candidates={mean_candidates(pred):.2f}"
        )

    print()
    print("=" * 90)
    print("RRF")
    print("=" * 90)

    evaluate_rrf(
        "dense local + char local",
        [
            dense_local,
            char_local,
        ],
        labels,
    )

    evaluate_rrf(
        "dense local + word local",
        [
            dense_local,
            word_local,
        ],
        labels,
    )

    evaluate_rrf(
        "dense local + char + word",
        [
            dense_local,
            char_local,
            word_local,
        ],
        labels,
    )

    # Dense standalone сильнее lexical,
    # поэтому сразу проверяем небольшой boost.
    evaluate_rrf(
        "dense(x1.5) + char + word",
        [
            dense_local,
            char_local,
            word_local,
        ],
        labels,
        weights=[
            1.5,
            1.0,
            1.0,
        ],
    )

    evaluate_rrf(
        "dense(x2) + char + word",
        [
            dense_local,
            char_local,
            word_local,
        ],
        labels,
        weights=[
            2.0,
            1.0,
            1.0,
        ],
    )

    print()
    print("=" * 90)
    print("POOL COVERAGE")
    print("=" * 90)

    evaluate_pool(
        "char + word",
        [
            char_local,
            word_local,
        ],
        labels,
    )

    evaluate_pool(
        "dense local + char",
        [
            dense_local,
            char_local,
        ],
        labels,
    )

    evaluate_pool(
        "dense local + word",
        [
            dense_local,
            word_local,
        ],
        labels,
    )

    evaluate_pool(
        "dense local + char + word",
        [
            dense_local,
            char_local,
            word_local,
        ],
        labels,
    )

    evaluate_pool(
        "local all + global dense",
        [
            dense_local,
            char_local,
            word_local,
            dense_global,
        ],
        labels,
    )


if __name__ == "__main__":
    main()

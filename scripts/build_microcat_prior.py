"""Строит leakage-safe nearest-query microcat prior для validation."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.microcat_prior import (
    MicrocatPrior,
)


TRAIN_PATH = Path("data/interim/train_part.parquet")

VAL_PATH = Path("data/interim/val_part.parquet")

OUTPUT_PATH = Path("artifacts/features/microcat_prior_val.pkl")


MODEL_NAME = "intfloat/multilingual-e5-base"

NEIGHBOR_K = 30
TOP_MICROCATS = 20

ENCODE_BATCH_SIZE = 128


def microcat_recall_at_k(
    true_microcats,
    priors,
    k,
):
    recalls = []

    for query_id, microcats in true_microcats.items():
        prior = priors.get(query_id)

        if prior is None:
            recalls.append(0.0)
            continue

        predicted = set(prior["microcats"][:k])

        if not microcats:
            continue

        hits = sum(microcat in predicted for microcat in microcats)

        recalls.append(hits / len(microcats))

    return float(np.mean(recalls))


def main():
    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    train_part = pd.read_parquet(TRAIN_PATH)

    val_part = pd.read_parquet(VAL_PATH)

    print(
        "Train rows:",
        len(train_part),
    )

    print(
        "Validation rows:",
        len(val_part),
    )

    # -----------------------------------------------------------------
    # Validation queries
    # -----------------------------------------------------------------

    val_queries = (
        val_part[
            [
                "query_signature",
                "search_query_norm",
                "search_infm_params_text",
            ]
        ]
        .drop_duplicates("query_signature")
        .set_index("query_signature")
    )

    print(
        "Validation signatures:",
        len(val_queries),
    )

    # -----------------------------------------------------------------
    # Fit history index
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILD MICROCAT HISTORY")
    print("=" * 80)

    prior_model = MicrocatPrior(
        model_name=MODEL_NAME,
        device="cuda",
        encode_batch_size=(ENCODE_BATCH_SIZE),
        max_seq_length=128,
        neighbor_k=NEIGHBOR_K,
        top_microcats=(TOP_MICROCATS),
        # Усиливаем самых близких соседей,
        # но пока без жесткого similarity threshold.
        similarity_power=4.0,
    )

    prior_model.fit(train_part)

    # -----------------------------------------------------------------
    # Predict validation priors
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("PREDICT VALIDATION PRIORS")
    print("=" * 80)

    priors = prior_model.predict(val_queries)

    # -----------------------------------------------------------------
    # Ground-truth microcats
    #
    # Дедуплицируем query-item, потому что Recall у нас
    # item-level, а повторные interaction rows не должны
    # увеличивать вес одного и того же positive.
    # -----------------------------------------------------------------

    truth = (
        val_part[
            [
                "query_signature",
                "item_id",
                "item_microcat_id",
            ]
        ]
        .dropna(subset=["item_microcat_id"])
        .drop_duplicates(
            [
                "query_signature",
                "item_id",
            ]
        )
    )

    truth["item_microcat_id"] = truth["item_microcat_id"].astype(np.int64)

    true_microcats = (
        truth.groupby("query_signature")["item_microcat_id"].apply(list).to_dict()
    )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("MICROCAT PRIOR QUALITY")
    print("=" * 80)

    for k in [
        1,
        3,
        5,
        10,
        20,
    ]:
        score = microcat_recall_at_k(
            true_microcats,
            priors,
            k=k,
        )

        print(f"Microcat Recall@{k}: {score:.6f}")

    top1_sims = np.asarray(
        [prior["neighbor_sim_top1"] for prior in priors.values()],
        dtype=np.float32,
    )

    mean5_sims = np.asarray(
        [prior["neighbor_sim_mean5"] for prior in priors.values()],
        dtype=np.float32,
    )

    print()
    print("Top-1 neighbor similarity:")

    print(f"  mean:   {top1_sims.mean():.6f}")

    print(f"  median: {np.median(top1_sims):.6f}")

    print(f"  p10:    {np.quantile(top1_sims, 0.10):.6f}")

    print(f"  p90:    {np.quantile(top1_sims, 0.90):.6f}")

    print()

    print("Mean top-5 neighbor similarity:")

    print(f"  mean:   {mean5_sims.mean():.6f}")

    print(f"  median: {np.median(mean5_sims):.6f}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "wb",
    ) as f:
        pickle.dump(
            priors,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

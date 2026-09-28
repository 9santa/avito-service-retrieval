import gc
import pickle
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.validation import normalize_query
from src.metrics import build_labels
from src.ranker_features import (
    RankerFeatureBuilder,
    FEATURE_NAMES,
)


DATA_DIR = "data/raw"

VAL_PATH = Path("data/interim/val_part.parquet")

PRED_DIR = Path("artifacts/predictions")

FEATURE_DIR = Path("artifacts/features")

MODEL_DIR = Path("artifacts/models")


# =====================================================================
# Validation artifacts
# =====================================================================

VAL_PRED_FILES = {
    "char_local": PRED_DIR / "char_local_val.pkl",
    "word_local": PRED_DIR / "word_local_val.pkl",
    "dense_local": PRED_DIR / "dense_local_val.pkl",
    "dense_global": PRED_DIR / "dense_val.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_top1_val.pkl",
    "dense_desc_local": PRED_DIR / "dense_desc_local_val.pkl",
    "dense_desc_global": PRED_DIR / "dense_desc_global_val.pkl",
}


VAL_SCORE_FILES = {
    "char_local": PRED_DIR / "char_local_scores_val.pkl",
    "word_local": PRED_DIR / "word_local_scores_val.pkl",
    "dense_local": PRED_DIR / "dense_local_scores_val.pkl",
    "dense_global": PRED_DIR / "dense_scores_val.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_top1_scores_val.pkl",
    "dense_desc_local": PRED_DIR / "dense_desc_local_scores_val.pkl",
    "dense_desc_global": PRED_DIR / "dense_desc_global_scores_val.pkl",
}


VAL_MICROCAT_PRIOR = FEATURE_DIR / "microcat_prior_val.pkl"


# =====================================================================
# Benchmark artifacts
# =====================================================================

BENCH_PRED_FILES = {
    "char_local": PRED_DIR / "char_local_bench.pkl",
    "word_local": PRED_DIR / "word_local_bench.pkl",
    "dense_local": PRED_DIR / "dense_local_bench.pkl",
    "dense_global": PRED_DIR / "dense_global_bench.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_bench.pkl",
    "dense_desc_local": PRED_DIR / "dense_desc_local_bench.pkl",
    "dense_desc_global": PRED_DIR / "dense_desc_global_bench.pkl",
}


BENCH_SCORE_FILES = {
    "char_local": PRED_DIR / "char_local_scores_bench.pkl",
    "word_local": PRED_DIR / "word_local_scores_bench.pkl",
    "dense_local": PRED_DIR / "dense_local_scores_bench.pkl",
    "dense_global": PRED_DIR / "dense_global_scores_bench.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_scores_bench.pkl",
    "dense_desc_local": PRED_DIR / "dense_desc_local_scores_bench.pkl",
    "dense_desc_global": PRED_DIR / "dense_desc_global_scores_bench.pkl",
}


BENCH_MICROCAT_PRIOR = FEATURE_DIR / "microcat_prior_bench.pkl"


META_MODEL_PATH = MODEL_DIR / "lgbm_rich_ranker.txt"

FINAL_MODEL_PATH = MODEL_DIR / "lgbm_final.txt"

ANSWER_PATH = Path("answer.csv")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dict(paths):
    result = {}

    for name, path in paths.items():
        print(f"Loading {name}: {path}")

        result[name] = load_pickle(path)

    return result


def validate_alignment(
    predictions,
    scores,
):
    for source in predictions:
        for query_id, items in list(predictions[source].items())[:1000]:
            source_scores = scores[source].get(
                query_id,
                [],
            )

            if len(items) != len(source_scores):
                raise ValueError(
                    f"{source}: "
                    f"{query_id}: "
                    f"{len(items)} items vs "
                    f"{len(source_scores)} scores"
                )

        print(f"{source}: alignment OK")


def main():
    # =================================================================
    # Data
    # =================================================================

    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    (
        train,
        benchmark_queries,
        benchmark_items,
    ) = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    train_items = build_train_item_corpus(train).reset_index(drop=True)

    benchmark_items = benchmark_items.reset_index(drop=True)

    if "search_query_norm" not in benchmark_queries.columns:
        benchmark_queries["search_query_norm"] = benchmark_queries["search_query"].map(
            normalize_query
        )

    # =================================================================
    # Validation data for FINAL ranker training
    # =================================================================

    val_queries = (
        val_part[
            [
                "query_signature",
                "search_query_norm",
                "search_infm_params_text",
                "search_location_id",
            ]
        ]
        .drop_duplicates("query_signature")
        .set_index("query_signature")
    )

    val_labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    # =================================================================
    # Load validation artifacts
    # =================================================================

    print()
    print("=" * 80)
    print("LOAD VALIDATION SOURCES")
    print("=" * 80)

    val_predictions = load_dict(VAL_PRED_FILES)

    val_scores = load_dict(VAL_SCORE_FILES)

    val_priors = load_pickle(VAL_MICROCAT_PRIOR)

    validate_alignment(
        val_predictions,
        val_scores,
    )

    # =================================================================
    # Build full final training matrix
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD FINAL TRAIN MATRIX")
    print("=" * 80)

    val_builder = RankerFeatureBuilder(
        items=train_items,
        queries=val_queries,
        predictions=(val_predictions),
        retrieval_scores=(val_scores),
        microcat_priors=(val_priors),
        missing_rank=51,
        rrf_k=60,
    )

    final_train = val_builder.build_matrix(
        query_ids=(val_queries.index.tolist()),
        labels=val_labels,
        # LambdaRank: only groups with >=1 positive
        skip_groups_without_positive=True,
        desc="Final train features",
    )

    X_train = final_train["X"]

    y_train = final_train["y"]

    train_group = final_train["group"]

    print(
        "X_train:",
        X_train.shape,
    )

    print(
        "Train groups:",
        len(train_group),
    )

    print(
        "Positive rows:",
        int(y_train.sum()),
    )

    # =================================================================
    # Use best iteration from last meta-validation model
    # =================================================================

    if META_MODEL_PATH.exists():
        meta_booster = lgb.Booster(model_file=str(META_MODEL_PATH))

        n_estimators = meta_booster.current_iteration()

        del meta_booster

        if n_estimators <= 0:
            n_estimators = 250
    else:
        n_estimators = 250

    print(
        "Final n_estimators:",
        n_estimators,
    )

    # =================================================================
    # Final LambdaRank
    # =================================================================

    print()
    print("=" * 80)
    print("TRAIN FINAL RANKER")
    print("=" * 80)

    ranker = LGBMRanker(
        objective="lambdarank",
        lambdarank_truncation_level=53,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=2.0,
        reg_alpha=0.0,
        colsample_bytree=0.9,
        subsample=0.9,
        subsample_freq=1,
        max_bin=127,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_group,
    )

    ranker.booster_.save_model(str(FINAL_MODEL_PATH))

    # Release millions of feature rows
    del X_train
    del y_train
    del train_group
    del final_train
    del val_builder
    del val_predictions
    del val_scores
    del val_priors

    gc.collect()

    # =================================================================
    # Benchmark queries
    # =================================================================

    print()
    print("=" * 80)
    print("LOAD BENCHMARK SOURCES")
    print("=" * 80)

    bench_queries = benchmark_queries.drop_duplicates("query_id").set_index("query_id")

    bench_predictions = load_dict(BENCH_PRED_FILES)

    bench_scores = load_dict(BENCH_SCORE_FILES)

    bench_priors = load_pickle(BENCH_MICROCAT_PRIOR)

    validate_alignment(
        bench_predictions,
        bench_scores,
    )

    # =================================================================
    # Benchmark features
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD BENCHMARK MATRIX")
    print("=" * 80)

    bench_builder = RankerFeatureBuilder(
        items=benchmark_items,
        queries=bench_queries,
        predictions=(bench_predictions),
        retrieval_scores=(bench_scores),
        microcat_priors=(bench_priors),
        missing_rank=51,
        rrf_k=60,
    )

    bench_query_ids = bench_queries.index.tolist()

    bench_data = bench_builder.build_matrix(
        query_ids=bench_query_ids,
        # labels unknown
        labels={},
        skip_groups_without_positive=False,
        desc="Benchmark features",
    )

    X_bench = bench_data["X"]

    bench_group = bench_data["group"]

    print(
        "X_bench:",
        X_bench.shape,
    )

    print(
        "Benchmark groups:",
        len(bench_group),
    )

    # =================================================================
    # Final predictions
    # =================================================================

    print()
    print("=" * 80)
    print("PREDICT")
    print("=" * 80)

    scores = ranker.predict(X_bench)

    predictions = bench_builder.scores_to_predictions(
        query_ids=(bench_data["query_ids"]),
        group_sizes=(bench_group),
        scores=scores,
        all_query_ids=(bench_query_ids),
        k=50,
    )

    # =================================================================
    # answer.csv
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD ANSWER.CSV")
    print("=" * 80)

    valid_item_ids = set(benchmark_items["item_id"].astype(str).str.lower())

    answers = []

    for query_id in benchmark_queries["query_id"].tolist():
        items = predictions.get(query_id, [])

        clean = []
        seen = set()

        for item_id in items:
            item_id = str(item_id).lower()

            if item_id in seen:
                continue

            if item_id not in valid_item_ids:
                raise ValueError(f"Unknown item_id: {item_id}")

            seen.add(item_id)
            clean.append(item_id)

            if len(clean) == 50:
                break

        answers.append(" ".join(clean))

    answer = pd.DataFrame(
        {
            "query_id": benchmark_queries["query_id"].tolist(),
            "answer": answers,
        }
    )

    # =================================================================
    # Submission validation
    # =================================================================

    if len(answer) != len(benchmark_queries):
        raise ValueError("Wrong number of rows")

    if answer["query_id"].duplicated().any():
        raise ValueError("Duplicate query_id")

    hex_re = re.compile(r"^[0-9a-f]{16}$")

    for row in answer.itertuples(index=False):
        ids = row.answer.split() if row.answer else []

        if len(ids) > 50:
            raise ValueError(f"{row.query_id}: >50 items")

        if len(ids) != len(set(ids)):
            raise ValueError(f"{row.query_id}: duplicates")

        for item_id in ids:
            if not hex_re.fullmatch(item_id):
                raise ValueError(f"Invalid item_id format: {item_id}")

    answer.to_csv(
        ANSWER_PATH,
        index=False,
    )

    print()
    print(
        "Saved:",
        ANSWER_PATH.resolve(),
    )

    print(
        "Rows:",
        len(answer),
    )

    lengths = answer["answer"].str.split().str.len()

    print(
        "Min items/query:",
        lengths.min(),
    )

    print(
        "Mean items/query:",
        lengths.mean(),
    )

    print(
        "Max items/query:",
        lengths.max(),
    )

    print()
    print(answer.head())


if __name__ == "__main__":
    main()

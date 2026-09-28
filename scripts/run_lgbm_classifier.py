import gc
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit

from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.fusion import (
    rrf_fusion,
    union_predictions,
)
from src.metrics import (
    build_labels,
    recall_at_k,
)
from src.ranker_features import (
    RankerFeatureBuilder,
    FEATURE_NAMES,
)


# =====================================================================
# Paths
# =====================================================================

DATA_DIR = "data/raw"

VAL_PATH = Path("data/interim/val_part.parquet")

PRED_DIR = Path("artifacts/predictions")

FEATURE_DIR = Path("artifacts/features")

MODEL_DIR = Path("artifacts/models")


PREDICTION_FILES = {
    "char_local": PRED_DIR / "char_local_val.pkl",
    "word_local": PRED_DIR / "word_local_val.pkl",
    "dense_local": PRED_DIR / "dense_local_val.pkl",
    "dense_global": PRED_DIR / "dense_val.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_top1_val.pkl",
}


SCORE_FILES = {
    "char_local": PRED_DIR / "char_local_scores_val.pkl",
    "word_local": PRED_DIR / "word_local_scores_val.pkl",
    "dense_local": PRED_DIR / "dense_local_scores_val.pkl",
    "dense_global": PRED_DIR / "dense_scores_val.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_top1_scores_val.pkl",
}


MICROCAT_PRIOR_PATH = FEATURE_DIR / "microcat_prior_val.pkl"


MODEL_PATH = MODEL_DIR / "lgbm_classifier.txt"

OUTPUT_PATH = PRED_DIR / "lgbm_classifier_val.pkl"


# =====================================================================
# Meta split
# =====================================================================

META_TEST_SIZE = 0.30
META_RANDOM_STATE = 123


# =====================================================================
# Helpers
# =====================================================================


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def subset_predictions(
    predictions,
    query_ids,
):
    return {
        query_id: predictions.get(
            query_id,
            [],
        )
        for query_id in query_ids
    }


def print_recall_table(
    labels,
    predictions,
    name,
):
    print(name)

    for k in [10, 20, 50]:
        score = recall_at_k(
            labels,
            predictions,
            k=k,
        )

        print(f"  Recall@{k}: {score:.6f}")

    print()


def make_query_balanced_weights(
    group_sizes,
):
    """
    Каждая query имеет одинаковый суммарный вес = 1.

    Query с 100 candidates:
        вес каждой строки = 1 / 100

    Query с 160 candidates:
        вес каждой строки = 1 / 160
    """

    total_rows = int(group_sizes.sum())

    weights = np.empty(
        total_rows,
        dtype=np.float32,
    )

    offset = 0

    for group_size in group_sizes:
        group_size = int(group_size)

        end = offset + group_size

        weights[offset:end] = 1.0 / group_size

        offset = end

    return weights


def validate_score_alignment(
    predictions,
    retrieval_scores,
    max_queries=1000,
):
    print()
    print("=" * 80)
    print("CHECK SCORE ALIGNMENT")
    print("=" * 80)

    for source_name in predictions:
        if source_name not in retrieval_scores:
            continue

        checked = 0

        for query_id, items in predictions[source_name].items():
            scores = retrieval_scores[source_name].get(
                query_id,
                [],
            )

            if len(items) != len(scores):
                raise ValueError(
                    f"{source_name}: mismatch "
                    f"for {query_id}: "
                    f"{len(items)} items != "
                    f"{len(scores)} scores"
                )

            checked += 1

            if checked >= max_queries:
                break

        print(f"{source_name:<20} OK")


# =====================================================================
# Main
# =====================================================================


def main():
    # =================================================================
    # Load data
    # =================================================================

    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    train, _, _ = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    items = build_train_item_corpus(train).reset_index(drop=True)

    queries = (
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

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    print(
        "Corpus items:",
        len(items),
    )

    print(
        "Validation signatures:",
        len(queries),
    )

    # =================================================================
    # Load candidate sources
    # =================================================================

    print()
    print("=" * 80)
    print("LOAD RETRIEVAL SOURCES")
    print("=" * 80)

    predictions = {}

    for name, path in PREDICTION_FILES.items():
        print(f"Loading {name}: {path}")

        predictions[name] = load_pickle(path)

    # =================================================================
    # Load raw scores
    # =================================================================

    print()
    print("=" * 80)
    print("LOAD RETRIEVAL SCORES")
    print("=" * 80)

    retrieval_scores = {}

    for name, path in SCORE_FILES.items():
        print(f"Loading {name}: {path}")

        retrieval_scores[name] = load_pickle(path)

    validate_score_alignment(
        predictions,
        retrieval_scores,
    )

    # =================================================================
    # Microcat priors
    # =================================================================

    print()
    print("=" * 80)
    print("LOAD MICROCAT PRIOR")
    print("=" * 80)

    microcat_priors = load_pickle(MICROCAT_PRIOR_PATH)

    print(
        "Microcat priors:",
        len(microcat_priors),
    )

    # =================================================================
    # Same meta split as LambdaRank
    # =================================================================

    print()
    print("=" * 80)
    print("META SPLIT")
    print("=" * 80)

    signature_table = (
        val_part[
            [
                "query_signature",
                "search_query_norm",
            ]
        ]
        .drop_duplicates("query_signature")
        .reset_index(drop=True)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=META_TEST_SIZE,
        random_state=META_RANDOM_STATE,
    )

    train_idx, eval_idx = next(
        splitter.split(
            signature_table,
            groups=signature_table["search_query_norm"],
        )
    )

    meta_train_query_ids = signature_table.iloc[train_idx]["query_signature"].tolist()

    meta_eval_query_ids = signature_table.iloc[eval_idx]["query_signature"].tolist()

    print(
        "Meta train signatures:",
        len(meta_train_query_ids),
    )

    print(
        "Meta eval signatures:",
        len(meta_eval_query_ids),
    )

    print(
        "Meta train normalized queries:",
        signature_table.iloc[train_idx]["search_query_norm"].nunique(),
    )

    print(
        "Meta eval normalized queries:",
        signature_table.iloc[eval_idx]["search_query_norm"].nunique(),
    )

    eval_labels = {query_id: labels[query_id] for query_id in meta_eval_query_ids}

    # =================================================================
    # Same-meta-eval baselines
    # =================================================================

    print()
    print("=" * 80)
    print("SAME META-EVAL BASELINES")
    print("=" * 80)

    eval_sources = [
        subset_predictions(
            predictions[source],
            meta_eval_query_ids,
        )
        for source in [
            "char_local",
            "word_local",
            "dense_local",
            "dense_global",
            "dense_microcat",
        ]
    ]

    rrf_all = rrf_fusion(
        eval_sources,
        k=50,
        rrf_k=60,
    )

    print_recall_table(
        eval_labels,
        rrf_all,
        "5-source RRF",
    )

    eval_pool = union_predictions(eval_sources)

    pool_recall = recall_at_k(
        eval_labels,
        eval_pool,
        k=1000,
    )

    pool_sizes = np.asarray([len(x) for x in eval_pool.values()])

    print(
        "5-source pool recall:",
        f"{pool_recall:.6f}",
    )

    print(
        "Mean pool size:",
        f"{pool_sizes.mean():.2f}",
    )

    del eval_pool
    del eval_sources
    gc.collect()

    # =================================================================
    # Feature builder
    # =================================================================

    print()
    print("=" * 80)
    print("FEATURE BUILDER")
    print("=" * 80)

    builder = RankerFeatureBuilder(
        items=items,
        queries=queries,
        predictions=predictions,
        retrieval_scores=(retrieval_scores),
        microcat_priors=(microcat_priors),
        missing_rank=51,
        rrf_k=60,
    )

    print(
        "Features:",
        len(FEATURE_NAMES),
    )

    # =================================================================
    # Build classifier train matrix
    #
    # IMPORTANT:
    # LambdaRank удалял groups без positive.
    #
    # Binary classifier их ОСТАВЛЯЕТ:
    # все candidates в них являются валидными negatives.
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD CLASSIFIER TRAIN MATRIX")
    print("=" * 80)

    train_data = builder.build_matrix(
        query_ids=meta_train_query_ids,
        labels=labels,
        # Отличие от ranker:
        skip_groups_without_positive=False,
        desc="Classifier train",
    )

    X_train = train_data["X"]
    y_train = train_data["y"]
    train_group = train_data["group"]

    print()
    print(
        "X_train:",
        X_train.shape,
    )

    print(
        "Positive rows:",
        int(y_train.sum()),
    )

    print(
        "Negative rows:",
        int(len(y_train) - y_train.sum()),
    )

    print(
        "Train groups:",
        len(train_group),
    )

    print(
        "Mean group size:",
        f"{train_group.mean():.2f}",
    )

    print(
        "Positive rate:",
        f"{y_train.mean():.6f}",
    )

    # =================================================================
    # Query-balanced weights
    # =================================================================

    train_weights = make_query_balanced_weights(train_group)

    print(
        "Total train weight:",
        f"{train_weights.sum():.2f}",
    )

    print(
        "Expected ~ number of queries:",
        len(train_group),
    )

    # =================================================================
    # Train binary classifier
    #
    # Пока БЕЗ early stopping:
    # binary_logloss/AUC тоже не совпадают с Recall@50.
    #
    # Первый чистый эксперимент:
    # same features + binary objective.
    # =================================================================

    print()
    print("=" * 80)
    print("TRAIN LIGHTGBM CLASSIFIER")
    print("=" * 80)

    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=300,
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

    classifier.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
    )

    print("Classifier trained.")

    # =================================================================
    # Release TRAIN memory before building eval
    # =================================================================

    del X_train
    del y_train
    del train_weights
    del train_data

    gc.collect()

    # =================================================================
    # Build eval matrix
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD EVAL MATRIX")
    print("=" * 80)

    eval_data = builder.build_matrix(
        query_ids=meta_eval_query_ids,
        labels=labels,
        skip_groups_without_positive=False,
        desc="Classifier eval",
    )

    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    eval_group = eval_data["group"]

    print()
    print(
        "X_eval:",
        X_eval.shape,
    )

    print(
        "Positive rows:",
        int(y_eval.sum()),
    )

    print(
        "Eval groups:",
        len(eval_group),
    )

    print(
        "Mean group size:",
        f"{eval_group.mean():.2f}",
    )

    # =================================================================
    # Predict relevance probabilities
    # =================================================================

    print()
    print("=" * 80)
    print("CLASSIFIER EVALUATION")
    print("=" * 80)

    eval_scores = classifier.predict_proba(X_eval)[:, 1]

    classifier_predictions = builder.scores_to_predictions(
        query_ids=eval_data["query_ids"],
        group_sizes=eval_group,
        scores=eval_scores,
        all_query_ids=(meta_eval_query_ids),
        k=50,
    )

    print_recall_table(
        eval_labels,
        classifier_predictions,
        "Query-balanced LightGBM classifier",
    )

    # =================================================================
    # Score diagnostics
    # =================================================================

    positive_scores = eval_scores[y_eval == 1]

    negative_scores = eval_scores[y_eval == 0]

    print(
        "Mean positive score:",
        f"{positive_scores.mean():.6f}",
    )

    print(
        "Mean negative score:",
        f"{negative_scores.mean():.6f}",
    )

    # =================================================================
    # Feature importance
    # =================================================================

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    booster = classifier.booster_

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    )

    importance["gain_pct"] = (
        importance["gain"]
        / max(
            importance["gain"].sum(),
            1e-12,
        )
        * 100
    )

    importance = importance.sort_values(
        "gain_pct",
        ascending=False,
    ).reset_index(drop=True)

    print(
        importance[
            [
                "feature",
                "gain_pct",
                "split",
            ]
        ].to_string(
            index=False,
            float_format=(lambda x: f"{x:.3f}"),
        )
    )

    # =================================================================
    # Save
    # =================================================================

    print()
    print("=" * 80)
    print("SAVE")
    print("=" * 80)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    booster.save_model(str(MODEL_PATH))

    with open(
        OUTPUT_PATH,
        "wb",
    ) as f:
        pickle.dump(
            classifier_predictions,
            f,
            protocol=(pickle.HIGHEST_PROTOCOL),
        )

    print(
        "Saved model:",
        MODEL_PATH,
    )

    print(
        "Saved predictions:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

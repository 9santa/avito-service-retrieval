"""Обучает rich LightGBM LambdaRank selector на union retrieval-кандидатов."""

import gc
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
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


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = "data/raw"

VAL_PATH = Path("data/interim/val_part.parquet")

PRED_DIR = Path("artifacts/predictions")

MODEL_DIR = Path("artifacts/models")

MODEL_PATH = MODEL_DIR / "lgbm_rich_ranker.txt"

OUTPUT_PATH = PRED_DIR / "lgbm_rich_val.pkl"


PREDICTION_FILES = {
    "char_local": PRED_DIR / "char_local_val.pkl",
    "word_local": PRED_DIR / "word_local_val.pkl",
    "dense_local": PRED_DIR / "dense_local_val.pkl",
    "dense_global": PRED_DIR / "dense_val.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_top1_val.pkl",
    "dense_desc_local": PRED_DIR / "dense_desc_local_val.pkl",
    "dense_desc_global": PRED_DIR / "dense_desc_global_val.pkl",
}


SCORE_FILES = {
    "char_local": PRED_DIR / "char_local_scores_val.pkl",
    "word_local": PRED_DIR / "word_local_scores_val.pkl",
    "dense_local": PRED_DIR / "dense_local_scores_val.pkl",
    "dense_global": PRED_DIR / "dense_scores_val.pkl",
    "dense_microcat": PRED_DIR / "dense_microcat_top1_scores_val.pkl",
    "dense_desc_local": PRED_DIR / "dense_desc_local_scores_val.pkl",
    "dense_desc_global": PRED_DIR / "dense_desc_global_scores_val.pkl",
}

MICROCAT_PRIOR_PATH = Path("artifacts/features/microcat_prior_val.pkl")


# ---------------------------------------------------------------------
# Meta split
# ---------------------------------------------------------------------

META_TEST_SIZE = 0.30
META_RANDOM_STATE = 123


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

    for k in [
        10,
        20,
        50,
    ]:
        score = recall_at_k(
            labels,
            predictions,
            k=k,
        )

        print(f"  Recall@{k}: {score:.6f}")

    print()


def main():
    # =================================================================
    # Load validation
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
    # Load retrieval predictions
    # =================================================================

    print()
    print("=" * 80)
    print("LOAD RETRIEVAL SOURCES")
    print("=" * 80)

    predictions = {}

    for name, path in PREDICTION_FILES.items():
        print(f"Loading {name}: {path}")

        predictions[name] = load_pickle(path)

    print()
    print("=" * 80)
    print("LOAD RETRIEVAL SCORES")
    print("=" * 80)

    retrieval_scores = {}

    for name, path in SCORE_FILES.items():
        print(f"Loading {name}: {path}")
        retrieval_scores[name] = load_pickle(path)

    # =================================================================
    # Load microcat prior
    # =================================================================

    microcat_priors = load_pickle(MICROCAT_PRIOR_PATH)

    # =================================================================
    # Meta train / eval split
    #
    # ВАЖНО:
    # split опять cold-query:
    # одинаковый search_query_norm не может попасть
    # одновременно в meta-train и meta-eval.
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
    # Same-eval baselines
    # =================================================================

    print()
    print("=" * 80)
    print("SAME META-EVAL BASELINES")
    print("=" * 80)

    char_eval = subset_predictions(
        predictions["char_local"],
        meta_eval_query_ids,
    )

    word_eval = subset_predictions(
        predictions["word_local"],
        meta_eval_query_ids,
    )

    dense_local_eval = subset_predictions(
        predictions["dense_local"],
        meta_eval_query_ids,
    )

    dense_global_eval = subset_predictions(
        predictions["dense_global"],
        meta_eval_query_ids,
    )

    dense_microcat_eval = subset_predictions(
        predictions["dense_microcat"],
        meta_eval_query_ids,
    )

    dense_desc_local_eval = subset_predictions(
        predictions["dense_desc_local"],
        meta_eval_query_ids,
    )

    dense_desc_global_eval = subset_predictions(
        predictions["dense_desc_global"],
        meta_eval_query_ids,
    )

    # Local 4-source RRF
    rrf_local = rrf_fusion(
        [
            dense_local_eval,
            char_eval,
            word_eval,
            dense_desc_local_eval,
        ],
        k=50,
        rrf_k=60,
    )

    print_recall_table(
        eval_labels,
        rrf_local,
        "Local dense + char + word RRF + dense desc local",
    )

    # All 7 source RRF
    rrf_all = rrf_fusion(
        [
            dense_local_eval,
            char_eval,
            word_eval,
            dense_global_eval,
            dense_microcat_eval,
            dense_desc_local_eval,
            dense_desc_global_eval,
        ],
        k=50,
        rrf_k=60,
    )

    print_recall_table(
        eval_labels,
        rrf_all,
        "All 7 sources RRF",
    )

    # Full pool
    eval_pool = union_predictions(
        [
            dense_local_eval,
            char_eval,
            word_eval,
            dense_global_eval,
            dense_microcat_eval,
            dense_desc_local_eval,
            dense_desc_global_eval,
        ]
    )

    pool_recall = recall_at_k(
        eval_labels,
        eval_pool,
        k=1000,
    )

    pool_sizes = np.asarray([len(items_) for items_ in eval_pool.values()])

    print(
        "All 7 full pool recall:",
        f"{pool_recall:.6f}",
    )

    print(
        "Mean eval pool size:",
        f"{pool_sizes.mean():.2f}",
    )

    # Освобождаем крупный eval_pool dict.
    del eval_pool
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
        retrieval_scores=retrieval_scores,
        microcat_priors=microcat_priors,
        missing_rank=51,
        rrf_k=60,
    )

    print(
        "Features:",
        len(FEATURE_NAMES),
    )

    # =================================================================
    # Build TRAIN
    #
    # Здесь группы без positive внутри candidate pool
    # выбрасываем.
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD TRAIN MATRIX")
    print("=" * 80)

    train_data = builder.build_matrix(
        query_ids=meta_train_query_ids,
        labels=labels,
        skip_groups_without_positive=True,
        desc="Train features",
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
        "Positive train rows:",
        int(y_train.sum()),
    )

    print(
        "Train groups:",
        len(train_group),
    )

    print(
        "Mean train group size:",
        train_group.mean(),
    )

    # =================================================================
    # Build EVAL
    #
    # На eval НЕ удаляем no-positive groups.
    # Иначе Recall будет завышен.
    # =================================================================

    print()
    print("=" * 80)
    print("BUILD EVAL MATRIX")
    print("=" * 80)

    eval_data = builder.build_matrix(
        query_ids=meta_eval_query_ids,
        labels=labels,
        skip_groups_without_positive=False,
        desc="Eval features",
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
        "Positive eval rows:",
        int(y_eval.sum()),
    )

    print(
        "Eval groups:",
        len(eval_group),
    )

    print(
        "Mean eval group size:",
        eval_group.mean(),
    )

    # =================================================================
    # LightGBM
    # =================================================================

    print()
    print("=" * 80)
    print("TRAIN LIGHTGBM")
    print("=" * 80)

    ranker = LGBMRanker(
        objective="lambdarank",
        # Наша target depth = 50.
        lambdarank_truncation_level=53,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=2.0,
        reg_alpha=0.0,
        colsample_bytree=0.9,
        # Реально включает row subsampling.
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
        eval_set=[
            (
                X_eval,
                y_eval,
            )
        ],
        eval_group=[eval_group],
        eval_at=[50],
        callbacks=[
            lgb.log_evaluation(period=20),
            lgb.early_stopping(
                stopping_rounds=50,
                first_metric_only=True,
            ),
        ],
    )

    print()
    print(
        "Best iteration:",
        ranker.best_iteration_,
    )

    # =================================================================
    # Predict
    # =================================================================

    print()
    print("=" * 80)
    print("RANKER EVALUATION")
    print("=" * 80)

    eval_scores = ranker.predict(
        X_eval,
        num_iteration=(ranker.best_iteration_),
    )

    ranker_predictions = builder.scores_to_predictions(
        query_ids=eval_data["query_ids"],
        group_sizes=eval_group,
        scores=eval_scores,
        # Инициализируем ВСЕ eval qids,
        # даже если вдруг у какого-то нет candidates.
        all_query_ids=(meta_eval_query_ids),
        k=50,
    )

    print_recall_table(
        eval_labels,
        ranker_predictions,
        "Rich LightGBM",
    )

    # =================================================================
    # Feature importance
    # =================================================================

    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    booster = ranker.booster_

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "gain": (booster.feature_importance(importance_type="gain")),
            "split": (booster.feature_importance(importance_type="split")),
        }
    )

    importance = importance.sort_values(
        "gain",
        ascending=False,
    ).reset_index(drop=True)

    importance["gain_pct"] = (
        importance["gain"]
        / max(
            importance["gain"].sum(),
            1e-12,
        )
        * 100
    )

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
            ranker_predictions,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
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

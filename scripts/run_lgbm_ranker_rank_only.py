"""Ранний rank-only LightGBM baseline для проверки ценности richer query-item features."""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.metrics import build_labels, recall_at_k
from src.fusion import rrf_fusion


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

VAL_PATH = Path("data/interim/val_part.parquet")

PRED_DIR = Path("artifacts/predictions")

PREDICTION_FILES = {
    "dense": PRED_DIR / "dense_val.pkl",
    "bm25": PRED_DIR / "bm25_val.pkl",
    "word": PRED_DIR / "word_tfidf_val.pkl",
    "char": PRED_DIR / "char_tfidf_val.pkl",
}

TOP_K = 50

META_EVAL_SIZE = 0.30
META_SPLIT_RANDOM_STATE = 123


TRAIN_SAMPLE_RANDOM_STATE = 42


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------


def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# Candidate features
# ---------------------------------------------------------------------

FEATURE_NAMES = [
    # Rank каждого retriever.
    "dense_rank",
    "bm25_rank",
    "word_rank",
    "char_rank",
    # Был ли item найден конкретным retriever.
    "dense_present",
    "bm25_present",
    "word_present",
    "char_present",
    # Aggregated rank-features.
    "num_sources",
    "best_rank",
    "mean_rank",
    "reciprocal_rank_sum",
]


def build_query_candidates(
    query_id,
    predictions,
):
    """
    Строит union top-50 кандидатов четырёх retriever'ов
    и rank-features для одного query_signature.

    Returns
    -------
    item_ids : list[str]
    features : np.ndarray, shape=(n_candidates, n_features)
    """
    sources = [
        "dense",
        "bm25",
        "word",
        "char",
    ]

    # item_id -> {source: rank}
    candidate_ranks = {}

    for source in sources:
        items = predictions[source].get(query_id, [])

        for rank, item_id in enumerate(items[:TOP_K], start=1):
            if item_id not in candidate_ranks:
                candidate_ranks[item_id] = {}

            candidate_ranks[item_id][source] = rank

    item_ids = list(candidate_ranks.keys())

    features = np.empty(
        (len(item_ids), len(FEATURE_NAMES)),
        dtype=np.float32,
    )

    missing_rank = TOP_K + 1

    for i, item_id in enumerate(item_ids):
        ranks = candidate_ranks[item_id]

        dense_rank = ranks.get("dense", missing_rank)
        bm25_rank = ranks.get("bm25", missing_rank)
        word_rank = ranks.get("word", missing_rank)
        char_rank = ranks.get("char", missing_rank)

        dense_present = float("dense" in ranks)
        bm25_present = float("bm25" in ranks)
        word_present = float("word" in ranks)
        char_present = float("char" in ranks)

        present_ranks = list(ranks.values())

        num_sources = len(present_ranks)
        best_rank = min(present_ranks)
        mean_rank = float(np.mean(present_ranks))

        reciprocal_rank_sum = sum(1.0 / rank for rank in present_ranks)

        features[i] = [
            dense_rank,
            bm25_rank,
            word_rank,
            char_rank,
            dense_present,
            bm25_present,
            word_present,
            char_present,
            num_sources,
            best_rank,
            mean_rank,
            reciprocal_rank_sum,
        ]

    return item_ids, features


def build_lgbm_dataset(
    query_ids,
    predictions,
    labels,
    drop_queries_without_positive,
):
    """
    Строит dataset для LightGBM Ranker.

    Важно:
    строки внутри одного query_signature идут подряд,
    а group_sizes сообщает LightGBM размеры query-групп.

    Parameters
    ----------
    query_ids
        Iterable query_signature.

    predictions
        dict[source -> dict[query_signature -> list[item_id]]]

    labels
        dict[query_signature -> set[item_id]]

    drop_queries_without_positive
        Для train=True разумно удалять query-группы,
        в candidate pool которых вообще нет positive.
        Они не дают LambdaRank ranking signal.

        Для evaluation=False — не удаляем ничего,
        чтобы Recall считался честно по всем запросам.

    Returns
    -------
    X   float32 matrix.

    y   uint8 relevance labels.

    group_sizes
        Размер candidate pool каждого запроса.

    kept_query_ids
        Query IDs в том же порядке, что group_sizes.

    candidates_by_query
        Список item_id для каждой query group.
        Нужен для восстановления predictions после predict().
    """
    X_parts = []
    y_parts = []

    group_sizes = []
    kept_query_ids = []
    candidates_by_query = []

    skipped_no_candidates = 0
    skipped_no_positive = 0

    for query_id in query_ids:
        item_ids, features = build_query_candidates(
            query_id=query_id,
            predictions=predictions,
        )

        if len(item_ids) == 0:
            skipped_no_candidates += 1
            continue

        positives = labels.get(query_id, set())

        y = np.fromiter(
            (1 if item_id in positives else 0 for item_id in item_ids),
            dtype=np.uint8,
            count=len(item_ids),
        )

        if drop_queries_without_positive and y.sum() == 0:
            skipped_no_positive += 1
            continue

        X_parts.append(features)
        y_parts.append(y)

        group_sizes.append(len(item_ids))
        kept_query_ids.append(query_id)
        candidates_by_query.append(item_ids)

    if not X_parts:
        raise RuntimeError("No candidate groups were created.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    group_sizes = np.asarray(
        group_sizes,
        dtype=np.int32,
    )

    print(f"Built dataset: {len(kept_query_ids)} queries, {len(y):,} candidate rows")

    print(
        "Positive rows:",
        int(y.sum()),
    )

    print(
        "Skipped queries without candidates:",
        skipped_no_candidates,
    )

    print(
        "Skipped train queries without positive in pool:",
        skipped_no_positive,
    )

    return (
        X,
        y,
        group_sizes,
        kept_query_ids,
        candidates_by_query,
    )


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------


def predict_top_k(
    model,
    X,
    group_sizes,
    query_ids,
    candidates_by_query,
    k=50,
):
    """
    Восстанавливает top-k item_id на каждый query после model.predict().
    """
    scores = model.predict(X)

    predictions = {}

    offset = 0

    for query_id, group_size, item_ids in zip(
        query_ids,
        group_sizes,
        candidates_by_query,
    ):
        query_scores = scores[offset : offset + group_size]

        offset += group_size

        if group_size <= k:
            order = np.argsort(query_scores)[::-1]
        else:
            top_idx = np.argpartition(
                query_scores,
                -k,
            )[-k:]

            order = top_idx[np.argsort(query_scores[top_idx])[::-1]]

        predictions[query_id] = [item_ids[i] for i in order]

    return predictions


def filter_labels(labels, query_ids):
    query_ids = set(query_ids)

    return {
        query_id: positives
        for query_id, positives in labels.items()
        if query_id in query_ids
    }


def union_pool_predictions(
    query_ids,
    predictions,
):
    """
    Полный union всех retriever top-50.
    Используется как upper bound текущего candidate pool.
    """
    result = {}

    for query_id in query_ids:
        seen = set()
        merged = []

        for source in [
            "dense",
            "bm25",
            "word",
            "char",
        ]:
            for item_id in predictions[source].get(query_id, []):
                if item_id not in seen:
                    seen.add(item_id)
                    merged.append(item_id)

        result[query_id] = merged

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    # -------------------------------------------------------------
    # 1. Load validation data
    # -------------------------------------------------------------

    val_part = pd.read_parquet(VAL_PATH)

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    meta = (
        val_part[
            [
                "query_signature",
                "search_query_norm",
            ]
        ]
        .drop_duplicates("query_signature")
        .reset_index(drop=True)
    )

    print("Validation signatures:", len(meta))
    print(
        "Validation normalized queries:",
        meta["search_query_norm"].nunique(),
    )

    # -------------------------------------------------------------
    # 2. Meta train/eval split
    #
    # Один search_query_norm целиком находится либо в LightGBM train,
    # либо в LightGBM eval.
    # -------------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=META_EVAL_SIZE,
        random_state=META_SPLIT_RANDOM_STATE,
    )

    train_idx, eval_idx = next(
        splitter.split(
            meta,
            groups=meta["search_query_norm"],
        )
    )

    lgbm_train_meta = meta.iloc[train_idx].reset_index(drop=True)

    lgbm_eval_meta = meta.iloc[eval_idx].reset_index(drop=True)

    assert set(lgbm_train_meta["search_query_norm"]).isdisjoint(
        set(lgbm_eval_meta["search_query_norm"])
    )

    print()
    print("Meta train signatures:", len(lgbm_train_meta))
    print("Meta eval signatures:", len(lgbm_eval_meta))

    print(
        "Meta train queries:",
        lgbm_train_meta["search_query_norm"].nunique(),
    )

    print(
        "Meta eval queries:",
        lgbm_eval_meta["search_query_norm"].nunique(),
    )

    # -------------------------------------------------------------
    # 4. Load retriever predictions
    # -------------------------------------------------------------

    predictions = {}

    for name, path in PREDICTION_FILES.items():
        print(f"Loading {name}: {path}")

        predictions[name] = load_predictions(path)

    # -------------------------------------------------------------
    # 5. Build LightGBM train dataset
    #
    # Группы без positive в pool удаляем:
    # LambdaRank не получает от них pairwise ranking signal.
    # -------------------------------------------------------------

    print()
    print("Building LightGBM train dataset...")

    (
        X_train,
        y_train,
        train_groups,
        train_query_ids,
        train_candidates,
    ) = build_lgbm_dataset(
        query_ids=lgbm_train_meta["query_signature"].tolist(),
        predictions=predictions,
        labels=labels,
        drop_queries_without_positive=True,
    )

    print()
    print("X_train:", X_train.shape)
    print("Train groups:", len(train_groups))

    # -------------------------------------------------------------
    # 6. Build evaluation dataset
    #
    # Здесь НЕ удаляем query без positive в pool:
    # иначе Recall будет искусственно завышен.
    # -------------------------------------------------------------

    print()
    print("Building LightGBM eval dataset...")

    (
        X_eval,
        y_eval,
        eval_groups,
        eval_query_ids,
        eval_candidates,
    ) = build_lgbm_dataset(
        query_ids=lgbm_eval_meta["query_signature"].tolist(),
        predictions=predictions,
        labels=labels,
        drop_queries_without_positive=False,
    )

    print()
    print("X_eval:", X_eval.shape)
    print("Eval groups:", len(eval_groups))

    # -------------------------------------------------------------
    # 7. Train LightGBM Ranker
    # -------------------------------------------------------------

    model = lgb.LGBMRanker(
        objective="lambdarank",
        lambdarank_truncation_level=53,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    print()
    print("Training LightGBM...")

    model.fit(
        X_train,
        y_train,
        group=train_groups,
        feature_name=FEATURE_NAMES,
    )

    # -------------------------------------------------------------
    # 8. Predict top-50
    # -------------------------------------------------------------

    lgbm_predictions = predict_top_k(
        model=model,
        X=X_eval,
        group_sizes=eval_groups,
        query_ids=eval_query_ids,
        candidates_by_query=eval_candidates,
        k=50,
    )

    eval_labels = filter_labels(
        labels,
        eval_query_ids,
    )

    # -------------------------------------------------------------
    # 9. LightGBM Recall@50
    # -------------------------------------------------------------

    lgbm_recall = recall_at_k(
        positives_by_query=eval_labels,
        predictions_by_query=lgbm_predictions,
        k=50,
    )

    # -------------------------------------------------------------
    # 10. Dense и Hybrid Weighted RRF baseline на том же meta-eval
    # -------------------------------------------------------------

    dense_predictions = {
        query_id: predictions["dense"].get(query_id, []) for query_id in eval_query_ids
    }

    dense_recall = recall_at_k(
        positives_by_query=eval_labels,
        predictions_by_query=dense_predictions,
        k=50,
    )

    rrf_all = rrf_fusion(
        [
            predictions["dense"],
            predictions["char"],
            predictions["word"],
            predictions["bm25"],
        ],
        weights=[
            2.0,
            1.0,
            1.0,
            1.0,
        ],
        k=50,
        rrf_k=60,
    )

    rrf_eval = {query_id: rrf_all[query_id] for query_id in eval_query_ids}

    rrf_recall = recall_at_k(
        positives_by_query=eval_labels,
        predictions_by_query=rrf_eval,
        k=50,
    )

    # -------------------------------------------------------------
    # 11. Full pool upper bound на том же meta-eval
    # -------------------------------------------------------------

    pool_predictions = union_pool_predictions(
        query_ids=eval_query_ids,
        predictions=predictions,
    )

    # Pool может содержать >50 items.
    # Берём k=1000, чтобы metric фактически оценивала весь union.
    pool_recall = recall_at_k(
        positives_by_query=eval_labels,
        predictions_by_query=pool_predictions,
        k=1000,
    )

    mean_pool_size = np.mean([len(items) for items in pool_predictions.values()])

    # -------------------------------------------------------------
    # 12. Results
    # -------------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Dense Recall@50:        {dense_recall:.6f}")
    print(f"Weighted RRF Recall@50: {rrf_recall:.6f}")
    print(f"LightGBM Recall@50:     {lgbm_recall:.6f}")
    print(f"Full pool recall:       {pool_recall:.6f}")
    print(f"Mean pool size:        {mean_pool_size:.2f}")

    # -------------------------------------------------------------
    # 13. Feature importance
    # -------------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("Feature importance:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()

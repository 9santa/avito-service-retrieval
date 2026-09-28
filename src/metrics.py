"""Метрики и построение ground-truth labels для retrieval-экспериментов."""

import pandas as pd


def recall_at_k(
    positives_by_query: dict,
    predictions_by_query: dict,
    k: int = 50,
):
    """
    Macro-avg Recall@K по запросам.

    positives_by_query:
        dict[query -> set[item_id]]

    predictions_by_query:
        dict[query -> list[item_id]]
    """
    recalls = []

    for query, positives in positives_by_query.items():
        if not positives:
            continue

        predicted = set(predictions_by_query.get(query, [])[:k])

        recall = len(positives & predicted) / len(positives)
        recalls.append(recall)

    return sum(recalls) / len(recalls)


def build_labels(
    df: pd.DataFrame,
    query_col="query_signature",
    item_col="item_id",
):
    """
    Собирает множество релевантных item_id для каждого запроса.
    """
    return df.groupby(query_col)[item_col].agg(set).to_dict()

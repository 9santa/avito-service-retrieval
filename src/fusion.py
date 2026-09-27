from collections import defaultdict


def rrf_fusion(
    prediction_lists: list[dict],
    k=50,
    rrf_k=60,
    weights=None,
):
    """
    Reciprocal Rank Fusion.

    prediction_lists:
        list[dict[query_id, list[item_id]]]

    weights:
        Вес каждого retriever. По умолчанию все равны.

    Returns:
        dict[query_id, list[item_id]]
    """
    if weights is None:
        weights = [1.0] * len(prediction_lists)

    all_queries = prediction_lists[0].keys()

    result = {}

    for query_id in all_queries:
        scores = defaultdict(float)

        for predictions, weight in zip(prediction_lists, weights):
            items = predictions.get(query_id, [])

            for rank, item_id in enumerate(items, start=1):
                scores[item_id] += weight / (rrf_k + rank)

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        result[query_id] = ranked[:k]

    return result


def union_predictions(prediction_lists: list[dict]):
    """
    Объединяет кандидатов всех retrievier'ов без ограничения top-k.
    """
    all_queries = prediction_lists[0].keys()

    result = {}

    for query_id in all_queries:
        seen = set()
        merged = []

        for predictions in prediction_lists:
            for item_id in predictions.get(query_id, []):
                if item_id not in seen:
                    seen.add(item_id)
                    merged.append(item_id)

        result[query_id] = merged

    return result


def quota_merge(
    local_predictions,
    global_predictions,
    local_k,
    total_k=50,
):
    """
    Берём до local_k кандидатов из local retrieval,
    затем заполняем оставшиеся места global retrieval.

    Дубликаты удаляются.
    """
    result = {}

    all_queries = set(local_predictions) | set(global_predictions)

    for query_id in all_queries:
        merged = []
        seen = set()

        # 1. Local quota
        for item_id in local_predictions.get(query_id, []):
            if item_id in seen:
                continue

            merged.append(item_id)
            seen.add(item_id)

            if len(merged) >= local_k:
                break

        # 2. Global заполнение до total_k
        for item_id in global_predictions.get(query_id, []):
            if item_id in seen:
                continue

            merged.append(item_id)
            seen.add(item_id)

            if len(merged) >= total_k:
                break

        result[query_id] = merged

    return result

import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.local_dense_retriever import (
    build_dense_query_text,
)
from src.fusion import union_predictions
from src.metrics import (
    build_labels,
    recall_at_k,
)


# =====================================================================
# Paths
# =====================================================================

DATA_DIR = "data/raw"

VAL_PATH = Path("data/interim/val_part.parquet")

EMBEDDING_DIR = Path("artifacts/embeddings")

PRED_DIR = Path("artifacts/predictions")


DESC_EMBEDDINGS_PATH = EMBEDDING_DIR / "e5_base_items_desc.npy"

DESC_ITEM_IDS_PATH = EMBEDDING_DIR / "e5_base_items_desc_item_ids.npy"


LOCAL_PRED_PATH = PRED_DIR / "dense_desc_local_val.pkl"

LOCAL_SCORE_PATH = PRED_DIR / "dense_desc_local_scores_val.pkl"

GLOBAL_PRED_PATH = PRED_DIR / "dense_desc_global_val.pkl"

GLOBAL_SCORE_PATH = PRED_DIR / "dense_desc_global_scores_val.pkl"


# Current 5 sources
CURRENT_SOURCE_PATHS = [
    PRED_DIR / "char_local_val.pkl",
    PRED_DIR / "word_local_val.pkl",
    PRED_DIR / "dense_local_val.pkl",
    PRED_DIR / "dense_val.pkl",
    PRED_DIR / "dense_microcat_top1_val.pkl",
]


# =====================================================================
# Config
# =====================================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

# Description делает sequences длиннее.
MAX_SEQ_LENGTH = 256

# Batch size внутри model.encode().
# Это ограничивает в первую очередь GPU memory во время encoder inference.
ENCODE_BATCH_SIZE = 64

# Query batching local exact search.
LOCAL_QUERY_BATCH_SIZE = 128

K = 50


# =====================================================================
# Text
# =====================================================================


def build_description_item_text(
    items: pd.DataFrame,
):
    """
    Первый clean ablation:

        title + params + description

    Не повторяем title, хотим измерить именно вклад description.
    """

    title = items["item_title_raw"].fillna("").astype(str)

    params = items["item_infm_params_text"].fillna("").astype(str)

    description = items["item_description_raw"].fillna("").astype(str)

    return "passage: " + title + " " + params + " " + description


# =====================================================================
# IO
# =====================================================================


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(
    obj,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "wb") as f:
        pickle.dump(
            obj,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# =====================================================================
# Item embeddings
# =====================================================================


def build_or_load_item_embeddings(
    model,
    items,
    item_ids,
):
    """
    Строим item embeddings целиком в RAM.

    Если cache уже существует и item IDs совпадают,
    просто загружаем обычный NumPy array в RAM.
    """

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DESC_EMBEDDINGS_PATH.exists() and DESC_ITEM_IDS_PATH.exists():
        cached_ids = np.load(
            DESC_ITEM_IDS_PATH,
            allow_pickle=True,
        )

        if np.array_equal(
            cached_ids,
            item_ids,
        ):
            print("Loading cached description embeddings:")

            print(DESC_EMBEDDINGS_PATH)

            return np.load(
                DESC_EMBEDDINGS_PATH,
            )

        print("Cached item IDs mismatch. Recomputing embeddings.")

    texts = build_description_item_text(items)

    print("Encoding description items...")

    embeddings = model.encode(
        texts.tolist(),
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(
        np.float32,
        copy=False,
    )

    # Сохраняем embeddings на диск
    np.save(
        DESC_EMBEDDINGS_PATH,
        embeddings,
    )

    np.save(
        DESC_ITEM_IDS_PATH,
        item_ids,
        allow_pickle=True,
    )

    print(
        "Saved embeddings:",
        DESC_EMBEDDINGS_PATH,
    )

    return embeddings


# =====================================================================
# Local exact search
# =====================================================================


def search_local_batch(
    query_embeddings,
    local_item_embeddings,
    k=50,
):
    """
    Exact cosine top-k для batch queries внутри одной location.

    Все embeddings L2 normalized,
    поэтому dot product == cosine.
    """

    n_queries = len(query_embeddings)

    n_items = len(local_item_embeddings)

    if n_items == 0:
        return (
            np.empty(
                (n_queries, 0),
                dtype=np.int32,
            ),
            np.empty(
                (n_queries, 0),
                dtype=np.float32,
            ),
        )

    k = min(
        k,
        n_items,
    )

    # -----------------------------------------------------------------
    # Exact cosine similarity
    # -----------------------------------------------------------------
    #
    # query_embeddings:
    #     (n_queries, embedding_dim)
    #
    # local_item_embeddings.T:
    #     (embedding_dim, n_items)
    #
    # scores:
    #     (n_queries, n_items)
    #
    # Так как embeddings нормализованы,
    # matrix multiplication == cosine similarity.
    # -----------------------------------------------------------------

    scores = query_embeddings @ local_item_embeddings.T

    # -----------------------------------------------------------------
    # Top-k для каждого query
    # -----------------------------------------------------------------

    if n_items <= k:
        # Если items меньше либо равно k,
        # просто полностью сортируем каждую строку.
        top_positions = np.argsort(
            scores,
            axis=1,
        )[:, ::-1]

    else:
        # argpartition быстро находит k лучших positions,
        # но не сортирует их между собой.
        top_positions = np.argpartition(
            scores,
            -k,
            axis=1,
        )[:, -k:]

        # Достаём scores только для найденных k кандидатов.
        top_scores_unsorted = np.take_along_axis(
            scores,
            top_positions,
            axis=1,
        )

        # Сортируем уже только k кандидатов по убыванию score
        order = np.argsort(
            top_scores_unsorted,
            axis=1,
        )[:, ::-1]

        top_positions = np.take_along_axis(
            top_positions,
            order,
            axis=1,
        )

    top_scores = np.take_along_axis(
        scores,
        top_positions,
        axis=1,
    )

    return (
        top_positions,
        top_scores,
    )


def retrieve_local(
    queries,
    query_embeddings,
    items,
    item_ids,
    item_embeddings,
):
    print()
    print("=" * 80)
    print("LOCAL DENSE + DESCRIPTION")
    print("=" * 80)

    # Для каждой location заранее сохраняем row indices items.
    #
    # Например:
    #     location A -> [0, 3, 8, 10]
    #     location B -> [1, 2, 5, 7]
    #
    # Это routing: query сравнивается только с items
    # той же location.
    location_to_item_indices = {
        location: group.index.to_numpy(dtype=np.int32)
        for location, group in items.groupby(
            "item_location_id",
            dropna=True,
        )
    }

    query_ids = queries.index.to_numpy()

    query_locations = queries["search_location_id"].to_numpy()

    location_to_query_positions = defaultdict(list)

    predictions = {}
    prediction_scores = {}

    # -----------------------------------------------------------------
    # Группируем query positions по location.
    # -----------------------------------------------------------------

    for pos, location in enumerate(query_locations):
        query_id = query_ids[pos]

        if pd.isna(location):
            predictions[query_id] = []

            prediction_scores[query_id] = []

            continue

        location_to_query_positions[location].append(pos)

    # -----------------------------------------------------------------
    # Exact search location-by-location.
    # -----------------------------------------------------------------

    for location, positions in tqdm(
        location_to_query_positions.items(),
        desc="Local retrieval",
    ):
        item_indices = location_to_item_indices.get(location)

        if item_indices is None or len(item_indices) == 0:
            for pos in positions:
                query_id = query_ids[pos]

                predictions[query_id] = []

                prediction_scores[query_id] = []

            continue

        # (n_local_items, embedding_dim)
        local_item_embeddings = np.ascontiguousarray(
            item_embeddings[item_indices],
            dtype=np.float32,
        )

        positions = np.asarray(
            positions,
            dtype=np.int32,
        )

        # Query batching нужен, чтобы не строить сразу огромную матрицу:
        # (all_queries_in_location x all_items_in_location)
        for start in range(
            0,
            len(positions),
            LOCAL_QUERY_BATCH_SIZE,
        ):
            end = min(
                start + LOCAL_QUERY_BATCH_SIZE,
                len(positions),
            )

            batch_positions = positions[start:end]

            q_emb = query_embeddings[batch_positions]

            # Возвращаются positions внутри local_item_embeddings,
            # а не глобальные rows корпуса.
            local_rows, scores = search_local_batch(
                query_embeddings=q_emb,
                local_item_embeddings=local_item_embeddings,
                k=K,
            )

            # Переводим local positions обратно
            # в глобальные row indices items.
            global_rows = item_indices[local_rows]

            for i, query_pos in enumerate(batch_positions):
                query_id = query_ids[query_pos]

                selected_rows = global_rows[i]

                selected_scores = scores[i]

                predictions[query_id] = item_ids[selected_rows].tolist()

                prediction_scores[query_id] = selected_scores.astype(float).tolist()

    return (
        predictions,
        prediction_scores,
    )


# =====================================================================
# Global exact FAISS search
# =====================================================================


def retrieve_global(
    queries,
    semantic_codes,
    unique_query_embeddings,
    item_ids,
    item_embeddings,
):
    print()
    print("=" * 80)
    print("GLOBAL DENSE + DESCRIPTION")
    print("=" * 80)

    dim = item_embeddings.shape[1]

    # IndexFlatIP делает exact inner-product search.
    # Так как item/query embeddings L2 normalized,
    # inner product == cosine similarity.
    index = faiss.IndexFlatIP(dim)

    print("Building exact FAISS index...")

    item_embeddings_faiss = np.ascontiguousarray(
        item_embeddings,
        dtype=np.float32,
    )

    index.add(item_embeddings_faiss)

    print(
        "Index size:",
        index.ntotal,
    )

    # unique_query_embeddings уже содержит embeddings
    # только уникальных semantic queries.
    scores, rows = index.search(
        np.ascontiguousarray(
            unique_query_embeddings,
            dtype=np.float32,
        ),
        K,
    )

    query_ids = queries.index.to_numpy()

    predictions = {}
    prediction_scores = {}

    # semantic_codes переводит каждую исходную query signature
    # в соответствующий unique query embedding.
    for pos, query_id in enumerate(query_ids):
        semantic_idx = semantic_codes[pos]

        selected_rows = rows[semantic_idx]

        selected_scores = scores[semantic_idx]

        valid = selected_rows >= 0

        selected_rows = selected_rows[valid]

        selected_scores = selected_scores[valid]

        predictions[query_id] = item_ids[selected_rows].tolist()

        prediction_scores[query_id] = selected_scores.astype(float).tolist()

    del index

    return (
        predictions,
        prediction_scores,
    )


# =====================================================================
# Metrics
# =====================================================================


def evaluate_retriever(
    name,
    labels,
    predictions,
):
    print()
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

    sizes = np.asarray([len(x) for x in predictions.values()])

    print(
        "  Mean candidates:",
        f"{sizes.mean():.2f}",
    )


def evaluate_pool(
    name,
    labels,
    sources,
):
    pool = union_predictions(sources)

    recall = recall_at_k(
        labels,
        pool,
        k=1000,
    )

    sizes = np.asarray([len(x) for x in pool.values()])

    print(f"{name:<45} recall={recall:.6f} | mean_pool={sizes.mean():.2f}")

    return recall


# =====================================================================
# Main
# =====================================================================


def main():
    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    train, _, _ = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    items = build_train_item_corpus(train).reset_index(drop=True)

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

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    item_ids = items["item_id"].to_numpy()

    print(
        "Corpus items:",
        len(items),
    )

    print(
        "Validation signatures:",
        len(val_queries),
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    model.half()

    # -----------------------------------------------------------------
    # Description item embeddings
    # -----------------------------------------------------------------

    item_embeddings = build_or_load_item_embeddings(
        model,
        items,
        item_ids,
    )

    print(
        "Item embeddings:",
        item_embeddings.shape,
    )

    # -----------------------------------------------------------------
    # Query embeddings
    #
    # IMPORTANT:
    # exactly same query representation as corrected global/local dense:
    #
    # query + search_params
    # -----------------------------------------------------------------

    query_texts = build_dense_query_text(val_queries)

    semantic_codes, unique_texts = pd.factorize(
        query_texts,
        sort=False,
    )

    unique_texts = unique_texts.tolist()

    print(
        "Unique semantic queries:",
        len(unique_texts),
    )

    # Кодируем только уникальные semantic queries.
    unique_query_embeddings = model.encode(
        unique_texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(
        np.float32,
        copy=False,
    )

    # Local needs embedding for every signature.
    #
    # semantic_codes восстанавливает embedding
    # для каждой исходной строки val_queries.
    query_embeddings = unique_query_embeddings[semantic_codes]

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Local
    # -----------------------------------------------------------------

    (
        local_predictions,
        local_scores,
    ) = retrieve_local(
        queries=val_queries,
        query_embeddings=query_embeddings,
        items=items,
        item_ids=item_ids,
        item_embeddings=item_embeddings,
    )

    evaluate_retriever(
        "Dense description LOCAL",
        labels,
        local_predictions,
    )

    save_pickle(
        local_predictions,
        LOCAL_PRED_PATH,
    )

    save_pickle(
        local_scores,
        LOCAL_SCORE_PATH,
    )

    # -----------------------------------------------------------------
    # Global
    # -----------------------------------------------------------------

    (
        global_predictions,
        global_scores,
    ) = retrieve_global(
        queries=val_queries,
        semantic_codes=semantic_codes,
        unique_query_embeddings=unique_query_embeddings,
        item_ids=item_ids,
        item_embeddings=item_embeddings,
    )

    evaluate_retriever(
        "Dense description GLOBAL",
        labels,
        global_predictions,
    )

    save_pickle(
        global_predictions,
        GLOBAL_PRED_PATH,
    )

    save_pickle(
        global_scores,
        GLOBAL_SCORE_PATH,
    )

    # -----------------------------------------------------------------
    # Current 5-source pool
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("POOL COVERAGE")
    print("=" * 80)

    current_sources = [load_pickle(path) for path in CURRENT_SOURCE_PATHS]

    current_recall = evaluate_pool(
        "Current 5 sources",
        labels,
        current_sources,
    )

    local_desc_recall = evaluate_pool(
        "+ local dense description",
        labels,
        current_sources + [local_predictions],
    )

    global_desc_recall = evaluate_pool(
        "+ global dense description",
        labels,
        current_sources + [global_predictions],
    )

    both_recall = evaluate_pool(
        "+ local + global dense description",
        labels,
        current_sources
        + [
            local_predictions,
            global_predictions,
        ],
    )

    print()
    print("=" * 80)
    print("POOL GAINS")
    print("=" * 80)

    print(
        "Local desc gain:",
        f"{local_desc_recall - current_recall:+.6f}",
    )

    print(
        "Global desc gain:",
        f"{global_desc_recall - current_recall:+.6f}",
    )

    print(
        "Both desc gain:",
        f"{both_recall - current_recall:+.6f}",
    )

    print()
    print(
        "Saved:",
        LOCAL_PRED_PATH,
    )

    print(
        "Saved:",
        LOCAL_SCORE_PATH,
    )

    print(
        "Saved:",
        GLOBAL_PRED_PATH,
    )

    print(
        "Saved:",
        GLOBAL_SCORE_PATH,
    )


if __name__ == "__main__":
    main()

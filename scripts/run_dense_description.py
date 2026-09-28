import gc
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

ENCODE_BATCH_SIZE = 64

ITEM_ENCODE_CHUNK_SIZE = 2048

LOCAL_QUERY_BATCH_SIZE = 128
LOCAL_ITEM_CHUNK_SIZE = 20_000

GLOBAL_INDEX_ADD_BATCH = 20_000

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

    Не повторяем title и не вводим дополнительных tricks.
    Хотим измерить именно вклад description.
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
                mmap_mode="r",
            )

        print("Cached item IDs mismatch. Recomputing embeddings.")

    texts = build_description_item_text(items)

    dim = model.get_sentence_embedding_dimension()

    embeddings = np.lib.format.open_memmap(
        DESC_EMBEDDINGS_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(items),
            dim,
        ),
    )

    print("Encoding description items...")

    for start in tqdm(
        range(
            0,
            len(items),
            ITEM_ENCODE_CHUNK_SIZE,
        ),
        desc="Encoding items",
    ):
        end = min(
            start + ITEM_ENCODE_CHUNK_SIZE,
            len(items),
        )

        chunk_texts = texts.iloc[start:end].tolist()

        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(
            np.float32,
            copy=False,
        )

        embeddings[start:end] = chunk_embeddings

    embeddings.flush()

    del embeddings

    np.save(
        DESC_ITEM_IDS_PATH,
        item_ids,
        allow_pickle=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "Saved embeddings:",
        DESC_EMBEDDINGS_PATH,
    )

    return np.load(
        DESC_EMBEDDINGS_PATH,
        mmap_mode="r",
    )


# =====================================================================
# Local exact search
# =====================================================================


def search_local_batch(
    query_embeddings,
    item_indices,
    item_embeddings,
    k=50,
):
    """
    Exact cosine top-k.

    Все embeddings L2 normalized,
    поэтому dot product == cosine.
    """

    n_queries = len(query_embeddings)

    n_items = len(item_indices)

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

    best_scores = np.full(
        (n_queries, k),
        -np.inf,
        dtype=np.float32,
    )

    best_rows = np.full(
        (n_queries, k),
        -1,
        dtype=np.int32,
    )

    for start in range(
        0,
        n_items,
        LOCAL_ITEM_CHUNK_SIZE,
    ):
        end = min(
            start + LOCAL_ITEM_CHUNK_SIZE,
            n_items,
        )

        rows = item_indices[start:end]

        item_chunk = np.asarray(
            item_embeddings[rows],
            dtype=np.float32,
        )

        scores = query_embeddings @ item_chunk.T

        chunk_k = min(
            k,
            scores.shape[1],
        )

        if scores.shape[1] <= chunk_k:
            chunk_top = np.argsort(
                scores,
                axis=1,
            )[:, ::-1]
        else:
            chunk_top = np.argpartition(
                scores,
                -chunk_k,
                axis=1,
            )[:, -chunk_k:]

            unsorted_scores = np.take_along_axis(
                scores,
                chunk_top,
                axis=1,
            )

            order = np.argsort(
                unsorted_scores,
                axis=1,
            )[:, ::-1]

            chunk_top = np.take_along_axis(
                chunk_top,
                order,
                axis=1,
            )

        chunk_scores = np.take_along_axis(
            scores,
            chunk_top,
            axis=1,
        )

        chunk_rows = rows[chunk_top]

        candidate_scores = np.concatenate(
            [
                best_scores,
                chunk_scores,
            ],
            axis=1,
        )

        candidate_rows = np.concatenate(
            [
                best_rows,
                chunk_rows,
            ],
            axis=1,
        )

        keep = np.argpartition(
            candidate_scores,
            -k,
            axis=1,
        )[:, -k:]

        best_scores = np.take_along_axis(
            candidate_scores,
            keep,
            axis=1,
        )

        best_rows = np.take_along_axis(
            candidate_rows,
            keep,
            axis=1,
        )

    order = np.argsort(
        best_scores,
        axis=1,
    )[:, ::-1]

    best_scores = np.take_along_axis(
        best_scores,
        order,
        axis=1,
    )

    best_rows = np.take_along_axis(
        best_rows,
        order,
        axis=1,
    )

    return (
        best_rows,
        best_scores,
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

    for pos, location in enumerate(query_locations):
        query_id = query_ids[pos]

        if pd.isna(location):
            predictions[query_id] = []

            prediction_scores[query_id] = []

            continue

        location_to_query_positions[location].append(pos)

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

        positions = np.asarray(
            positions,
            dtype=np.int32,
        )

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

            rows, scores = search_local_batch(
                q_emb,
                item_indices,
                item_embeddings,
                k=K,
            )

            for i, query_pos in enumerate(batch_positions):
                query_id = query_ids[query_pos]

                valid = rows[i] >= 0

                selected_rows = rows[i][valid]

                selected_scores = scores[i][valid]

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

    index = faiss.IndexFlatIP(dim)

    print("Building exact FAISS index...")

    for start in tqdm(
        range(
            0,
            len(item_embeddings),
            GLOBAL_INDEX_ADD_BATCH,
        ),
        desc="FAISS add",
    ):
        end = min(
            start + GLOBAL_INDEX_ADD_BATCH,
            len(item_embeddings),
        )

        chunk = np.ascontiguousarray(
            item_embeddings[start:end],
            dtype=np.float32,
        )

        index.add(chunk)

    print(
        "Index size:",
        index.ntotal,
    )

    scores, rows = index.search(
        np.ascontiguousarray(unique_query_embeddings),
        K,
    )

    query_ids = queries.index.to_numpy()

    predictions = {}
    prediction_scores = {}

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
        query_embeddings=(query_embeddings),
        items=items,
        item_ids=item_ids,
        item_embeddings=(item_embeddings),
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
        semantic_codes=(semantic_codes),
        unique_query_embeddings=(unique_query_embeddings),
        item_ids=item_ids,
        item_embeddings=(item_embeddings),
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

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.local_dense_retriever import build_dense_query_text
from src.metrics import (
    build_labels,
    recall_at_k,
)


DATA_DIR = "data/raw"
VAL_PATH = Path("data/interim/val_part.parquet")

EMBEDDINGS_PATH = Path("artifacts/embeddings/e5_base_items.npy")

EMBEDDING_IDS_PATH = Path("artifacts/embeddings/e5_base_items_item_ids.npy")

PRED_PATH = Path("artifacts/predictions/dense_val.pkl")

SCORES_PATH = Path("artifacts/predictions/dense_scores_val.pkl")


MODEL_NAME = "intfloat/multilingual-e5-base"

ENCODE_BATCH_SIZE = 128
MAX_SEQ_LENGTH = 128

INDEX_ADD_BATCH_SIZE = 20_000
K = 50


def main():
    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    train, _, _ = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    items = build_train_item_corpus(train).reset_index(drop=True)

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

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    item_ids = items["item_id"].to_numpy()

    print("Corpus items:", len(items))
    print(
        "Validation signatures:",
        len(val_queries),
    )

    # ---------------------------------------------------------
    # Existing item embeddings
    # ---------------------------------------------------------

    item_embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    print(
        "Item embeddings:",
        item_embeddings.shape,
    )

    if len(item_embeddings) != len(items):
        raise ValueError("Embedding count != corpus size")

    if EMBEDDING_IDS_PATH.exists():
        cached_ids = np.load(
            EMBEDDING_IDS_PATH,
            allow_pickle=True,
        )

        if not np.array_equal(
            cached_ids,
            item_ids,
        ):
            raise ValueError("Item order does not match embedding cache.")

        print("Item ID alignment: OK")

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # semantic identity =
    # search_query_norm + search_infm_params_text
    #
    # build_dense_query_text also adds "query:".
    # ---------------------------------------------------------

    query_texts = build_dense_query_text(val_queries)

    codes, unique_texts = pd.factorize(
        query_texts,
        sort=False,
    )

    unique_texts = unique_texts.tolist()

    print(
        "Unique semantic queries:",
        len(unique_texts),
    )

    # ---------------------------------------------------------
    # Encode queries
    # ---------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    model.half()

    query_embeddings = model.encode(
        unique_texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "Query embeddings:",
        query_embeddings.shape,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Exact global FAISS index
    # ---------------------------------------------------------

    dim = item_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    print("Building FAISS index...")

    for start in range(
        0,
        len(item_embeddings),
        INDEX_ADD_BATCH_SIZE,
    ):
        end = min(
            start + INDEX_ADD_BATCH_SIZE,
            len(item_embeddings),
        )

        chunk = np.ascontiguousarray(
            item_embeddings[start:end],
            dtype=np.float32,
        )

        index.add(chunk)

    print(
        "FAISS items:",
        index.ntotal,
    )

    # ---------------------------------------------------------
    # Search unique queries
    # ---------------------------------------------------------

    scores, indices = index.search(
        np.ascontiguousarray(query_embeddings),
        K,
    )

    # ---------------------------------------------------------
    # Expand semantic queries back to signatures
    # ---------------------------------------------------------

    query_ids = val_queries.index.to_numpy()

    predictions = {}
    prediction_scores = {}

    for row_pos, query_id in enumerate(query_ids):
        semantic_idx = codes[row_pos]

        corpus_rows = indices[semantic_idx]

        row_scores = scores[semantic_idx]

        valid = corpus_rows >= 0

        corpus_rows = corpus_rows[valid]

        row_scores = row_scores[valid]

        predictions[query_id] = item_ids[corpus_rows].tolist()

        prediction_scores[query_id] = row_scores.astype(float).tolist()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    print()

    for k in [10, 20, 50]:
        score = recall_at_k(
            labels,
            predictions,
            k=k,
        )

        print(f"Recall@{k}: {score:.6f}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    PRED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PRED_PATH,
        "wb",
    ) as f:
        pickle.dump(
            predictions,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open(
        SCORES_PATH,
        "wb",
    ) as f:
        pickle.dump(
            prediction_scores,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print("Saved:", PRED_PATH)
    print("Saved:", SCORES_PATH)


if __name__ == "__main__":
    main()

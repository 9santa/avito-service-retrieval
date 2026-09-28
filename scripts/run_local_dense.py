"""Строит location-aware E5 dense кандидатов и сохраняет raw cosine scores."""

import pickle
from pathlib import Path

import pandas as pd


from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.local_dense_retriever import (
    LocalDenseRetriever,
)
from src.metrics import (
    build_labels,
    recall_at_k,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = "data/raw"

VAL_PATH = Path("data/interim/val_part.parquet")

ARTIFACT_DIR = Path("artifacts")

EMBEDDING_PATH = ARTIFACT_DIR / "embeddings" / "e5_base_items.npy"

PREDICTIONS_PATH = ARTIFACT_DIR / "predictions" / "dense_local_val.pkl"

SCORES_PATH = ARTIFACT_DIR / "predictions" / "dense_local_scores_val.pkl"


# ---------------------------------------------------------------------
# Dense config
# ---------------------------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-base"

ENCODE_BATCH_SIZE = 128

MAX_SEQ_LENGTH = 128

ITEM_ENCODE_CHUNK_SIZE = 4096

# Сколько queries одновременно участвуют
# в local matrix multiplication.
QUERY_BATCH_SIZE = 128


def main():
    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    train, _, _ = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    local_items = build_train_item_corpus(train).reset_index(drop=True)

    print(
        "Corpus items:",
        len(local_items),
    )

    # -----------------------------------------------------------------
    # Validation queries
    #
    # В отличие от global query-only retrieval здесь нельзя
    # дедуплицировать по search_query_norm:
    # location влияет на candidate set.
    # -----------------------------------------------------------------

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

    print(
        "Validation signatures:",
        len(val_queries),
    )

    print(
        "Validation locations:",
        val_queries["search_location_id"].nunique(),
    )

    # -----------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    # -----------------------------------------------------------------
    # Retriever
    # -----------------------------------------------------------------

    retriever = LocalDenseRetriever(
        model_name=MODEL_NAME,
        device="cuda",
        encode_batch_size=ENCODE_BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        fp16=True,
        query_batch_size=(QUERY_BATCH_SIZE),
    )

    # -----------------------------------------------------------------
    # Item embeddings
    # -----------------------------------------------------------------

    retriever.fit(
        items=local_items,
        embeddings_path=EMBEDDING_PATH,
        force_recompute=False,
    )

    print(
        "Embedding matrix:",
        retriever.item_embeddings.shape,
    )

    print(
        "Locations:",
        len(retriever.location_to_indices),
    )

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    predictions, prediction_scores = retriever.retrieve(queries=val_queries, k=50)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    print()

    for k in [
        10,
        20,
        50,
    ]:
        score = recall_at_k(
            positives_by_query=labels,
            predictions_by_query=predictions,
            k=k,
        )

        print(f"Recall@{k}: {score:.6f}")

    # -----------------------------------------------------------------
    # Candidate stats
    # -----------------------------------------------------------------

    candidate_counts = pd.Series([len(items) for items in predictions.values()])

    print()

    print(
        "Queries with <50 candidates:",
        int((candidate_counts < 50).sum()),
    )

    print(
        "Mean candidates:",
        candidate_counts.mean(),
    )

    print(
        "Median candidates:",
        candidate_counts.median(),
    )

    print(
        "Min candidates:",
        candidate_counts.min(),
    )

    # -----------------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------------

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PREDICTIONS_PATH,
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

    print(
        "Saved predictions:",
        PREDICTIONS_PATH,
    )

    print(
        "Saved scores:",
        SCORES_PATH,
    )


if __name__ == "__main__":
    main()

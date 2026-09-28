import pickle
from pathlib import Path

import pandas as pd


from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.dense_retriever import GlobalDenseRetriever
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

    print("Corpus items:", len(items))
    print(
        "Validation signatures:",
        len(val_queries),
    )

    # ---------------------------------------------------------
    # Global dense retrieval using existing item embeddings
    # ---------------------------------------------------------

    retriever = GlobalDenseRetriever(
        model_name=MODEL_NAME,
        batch_size=ENCODE_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LENGTH,
        device="cuda",
        fp16=True,
    )

    retriever.fit(
        items=items,
        embeddings_path=EMBEDDINGS_PATH,
        embedding_ids_path=EMBEDDING_IDS_PATH,
        index_add_batch_size=INDEX_ADD_BATCH_SIZE,
    )

    print("FAISS items:", retriever.index.ntotal)

    predictions, prediction_scores = retriever.retrieve(
        queries=val_queries,
        k=K,
        return_scores=True,
    )

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

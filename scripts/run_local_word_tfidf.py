"""Строит location-aware word TF-IDF кандидатов и сохраняет raw cosine scores."""

import pickle
from pathlib import Path

import pandas as pd


from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.local_retrieval import LocalWordTfidfRetriever
from src.metrics import build_labels, recall_at_k


DATA_DIR = "data/raw"
VAL_PATH = "data/interim/val_part.parquet"

OUTPUT_PATH = Path("artifacts/predictions/word_local_val.pkl")
SCORES_PATH = Path("artifacts/predictions/word_local_scores_val.pkl")


def main():
    train, _, _ = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    local_items = build_train_item_corpus(train).reset_index(drop=True)

    # Здесь нельзя дедуплицировать только по тексту запроса:
    # location участвует в retrieval
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

    print("Corpus items:", len(local_items))
    print("Validation signatures:", len(val_queries))

    retriever = LocalWordTfidfRetriever(
        ngram_range=(1, 2),
        min_df=2,
    )

    retriever.fit(local_items)

    print(
        "TF-IDF matrix:",
        retriever.item_matrix.shape,
    )

    print(
        "Vocabulary:",
        len(retriever.vectorizer.vocabulary_),
    )

    print(
        "Locations:",
        len(retriever.location_to_indices),
    )

    predictions, prediction_scores = retriever.retrieve(
        val_queries,
        k=50,
        return_scores=True,
    )

    print()

    for k in [10, 20, 50]:
        score = recall_at_k(
            labels,
            predictions,
            k=k,
        )

        print(f"Recall@{k}: {score:.6f}")

    counts = pd.Series([len(x) for x in predictions.values()])

    print()
    print(
        "Queries with <50 candidates:",
        int((counts < 50).sum()),
    )

    print(
        "Mean candidates:",
        counts.mean(),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(predictions, f)

    print()
    print("Saved:", OUTPUT_PATH)

    with open(SCORES_PATH, "wb") as f:
        pickle.dump(
            prediction_scores,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print(
        "Saved scores:",
        SCORES_PATH,
    )


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd
import pickle

from src.data import load_data, build_train_item_corpus
from src.metrics import build_labels, recall_at_k
from src.word_tfidf_retriever import WordTfidfRetriever


DATA_DIR = "data/raw"


def main():
    # 1. Загружаем данные
    train, _, _ = load_data(DATA_DIR)

    # 2. Загружаем cold-query validation split
    train_part = pd.read_parquet("data/interim/train_part.parquet")

    val_part = pd.read_parquet("data/interim/val_part.parquet")

    print(f"Train rows: {len(train_part)}")
    print(f"Validation rows: {len(val_part)}")

    # 3. Строим локальный корпус объявлений.
    # В корпус включаем все объявления из train, включая validation positives.
    # Имеем право: в реальной задаче на продакшене corpus тоже известен целиком.
    local_items = build_train_item_corpus(train)

    print(f"Local corpus items: {len(local_items)}")

    val_queries = (
        val_part[
            [
                "query_signature",
                "search_query_norm",
            ]
        ]
        .drop_duplicates("query_signature")
        .set_index("query_signature")
    )

    # Ground truth
    val_labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    print(f"Validation signatures: {len(val_queries)}")

    # Обучаем lexical retriever
    retriever = WordTfidfRetriever(
        ngram_range=(1, 2),
        min_df=2,
    )

    retriever.fit(local_items)

    print(
        "TF-IDF matrix:",
        retriever.item_matrix.shape,
    )

    print(
        "Vocabulary size:",
        len(retriever.vectorizer.vocabulary_),
    )

    # Candidate generation
    unique_val_queries = (
        val_part[["search_query_norm"]].drop_duplicates().reset_index(drop=True)
    )

    query_predictions = retriever.retrieve(
        unique_val_queries,
        query_col="search_query_norm",
        k=50,
        batch_size=256,
    )

    pred_by_text = {
        unique_val_queries.loc[idx, "search_query_norm"]: pred
        for idx, pred in query_predictions.items()
    }

    val_queries = val_part[
        [
            "query_signature",
            "search_query_norm",
        ]
    ].drop_duplicates("query_signature")

    predictions = {
        row.query_signature: pred_by_text[row.search_query_norm]
        for row in val_queries.itertuples()
    }

    # Save predictions
    Path("artifacts/predictions").mkdir(parents=True, exist_ok=True)

    with open(
        "artifacts/predictions/word_tfidf_val.pkl",
        mode="wb",
    ) as f:
        pickle.dump(predictions, f)

    # Metrics
    print()

    for k in [10, 20, 50]:
        score = recall_at_k(
            positives_by_query=val_labels,
            predictions_by_query=predictions,
            k=k,
        )

        print(f"Recall@{k}: {score:.6f}")

    # Sanity checks
    candidate_counts = pd.Series([len(x) for x in predictions.values()])

    print()
    print(
        "Queries with <50 candidates:",
        int((candidate_counts < 50).sum()),
    )

    print(
        "Mean candidates:",
        candidate_counts.mean(),
    )


if __name__ == "__main__":
    main()

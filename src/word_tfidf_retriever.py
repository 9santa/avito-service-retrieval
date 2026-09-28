"""Глобальный word TF-IDF baseline."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def build_item_text(items: pd.DataFrame):
    """
    Текст объявления для первого lexical baseline.
    title + параметры объявления.
    """
    title = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str)

    return title + " " + params


class WordTfidfRetriever:
    """Word unigram/bigram TF-IDF retriever для глобального lexical baseline."""

    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=2,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.item_matrix = None
        self.item_ids = None

    def fit(self, items):
        """Обучает vectorizer и индексирует корпус объявлений."""

        item_texts = build_item_text(items)

        self.item_matrix = self.vectorizer.fit_transform(item_texts)
        self.item_ids = items["item_id"].to_numpy()

        return self

    def retrieve(
        self,
        queries,
        query_col="search_query_norm",
        k=50,
        batch_size=256,
    ):
        """Возвращает top-k cosine TF-IDF кандидатов по каждой query row."""

        query_matrix = self.vectorizer.transform(queries[query_col].fillna(""))

        predictions = {}

        item_matrix_t = self.item_matrix.T.tocsr()

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))

            scores = (query_matrix[start:end] @ item_matrix_t).tocsr()

            for local_idx in range(end - start):
                row = scores.getrow(local_idx)

                if row.nnz == 0:
                    top_items = []

                else:
                    data = row.data
                    indices = row.indices

                    if len(data) > k:
                        top_pos = np.argpartition(
                            data,
                            -k,
                        )[-k:]

                        top_pos = top_pos[np.argsort(data[top_pos])[::-1]]
                    else:
                        top_pos = np.argsort(data)[::-1]

                    item_indices = indices[top_pos]

                    top_items = self.item_ids[item_indices].tolist()

                query_key = queries.index[start + local_idx]
                predictions[query_key] = top_items

        return predictions

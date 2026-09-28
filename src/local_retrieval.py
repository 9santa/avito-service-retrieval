import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Local = используем жесткую привязку по локации
"""


def build_word_item_text(items: pd.DataFrame):
    title = items["item_title_raw"].fillna("").astype(str)

    params = items["item_infm_params_text"].fillna("").astype(str)

    # Усиливаем title
    return title + " " + title + " " + params


def build_word_query_text(queries: pd.DataFrame):
    query = queries["search_query_norm"].fillna("").astype(str)

    params = queries["search_infm_params_text"].fillna("").astype(str)

    return query + " " + params


class LocalWordTfidfRetriever:
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
        self.location_to_indices = None

    def fit(self, items: pd.DataFrame):
        item_texts = build_word_item_text(items)

        self.item_matrix = self.vectorizer.fit_transform(item_texts).tocsr()

        self.item_ids = items["item_id"].to_numpy()

        # location -> row indices in item_matrix
        self.location_to_indices = {
            location: group.index.to_numpy()
            for location, group in (
                items.reset_index(drop=True).groupby("item_location_id")
            )
        }

        return self

    def retrieve(
        self,
        queries: pd.DataFrame,
        k=50,
        return_scores=False,
    ):
        query_texts = build_word_query_text(queries)

        query_matrix = self.vectorizer.transform(query_texts).tocsr()

        predictions = {}
        predictions_scores = {}

        for row_pos, (query_key, row) in enumerate(queries.iterrows()):
            location = row["search_location_id"]

            item_indices = self.location_to_indices.get(location)

            # Если в корпусе такой location вообще отсутствует.
            if item_indices is None or len(item_indices) == 0:
                predictions[query_key] = []
                predictions_scores[query_key] = []
                continue

            q = query_matrix[row_pos]

            local_matrix = self.item_matrix[item_indices]

            scores = (q @ local_matrix.T).tocsr()

            score_row = scores.getrow(0)

            if score_row.nnz == 0:
                predictions[query_key] = []
                predictions_scores[query_key] = []
                continue

            data = score_row.data
            local_cols = score_row.indices

            take_k = min(k, len(data))

            if len(data) > take_k:
                top_pos = np.argpartition(
                    data,
                    -take_k,
                )[-take_k:]

                top_pos = top_pos[np.argsort(data[top_pos])[::-1]]
            else:
                top_pos = np.argsort(data)[::-1]

            # local_cols — индексы внутри local_matrix.
            corpus_rows = item_indices[local_cols[top_pos]]

            selected_scores = data[top_pos]

            predictions[query_key] = self.item_ids[corpus_rows].tolist()
            predictions_scores[query_key] = selected_scores.astype(float).tolist()

        if return_scores:
            return (predictions, predictions_scores)

        return predictions


class LocalCharTfidfRetriever:
    def __init__(
        self,
        ngram_range=(3, 5),
        min_df=2,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            dtype=np.float32,
        )

        self.item_matrix = None
        self.item_ids = None
        self.location_to_indices = None

    def fit(self, items: pd.DataFrame):
        item_texts = items["item_title_raw"].fillna("").astype(str)

        self.item_matrix = self.vectorizer.fit_transform(item_texts).tocsr()

        self.item_ids = items["item_id"].to_numpy()

        self.location_to_indices = {
            location: group.index.to_numpy()
            for location, group in (
                items.reset_index(drop=True).groupby("item_location_id")
            )
        }

        return self

    def retrieve(
        self,
        queries,
        k=50,
        return_scores=False,
    ):
        query_matrix = self.vectorizer.transform(
            queries["search_query_norm"].fillna("").astype(str)
        ).tocsr()

        predictions = {}
        predictions_scores = {}

        for row_pos, (query_key, row) in enumerate(queries.iterrows()):
            location = row["search_location_id"]

            item_indices = self.location_to_indices.get(location)

            if item_indices is None or len(item_indices) == 0:
                predictions[query_key] = []
                predictions_scores[query_key] = []
                continue

            local_matrix = self.item_matrix[item_indices]

            scores = (query_matrix[row_pos] @ local_matrix.T).tocsr()

            score_row = scores.getrow(0)

            if score_row.nnz == 0:
                predictions[query_key] = []
                predictions_scores[query_key] = []
                continue

            data = score_row.data
            local_cols = score_row.indices

            take_k = min(k, len(data))

            if len(data) > take_k:
                top_pos = np.argpartition(
                    data,
                    -take_k,
                )[-take_k:]

                top_pos = top_pos[np.argsort(data[top_pos])[::-1]]
            else:
                top_pos = np.argsort(data)[::-1]

            corpus_rows = item_indices[local_cols[top_pos]]

            selected_scores = data[top_pos]

            predictions[query_key] = self.item_ids[corpus_rows].tolist()
            predictions_scores[query_key] = selected_scores.astype(float).tolist()

        if return_scores:
            return (predictions, predictions_scores)

        return predictions

"""Глобальный BM25 baseline по тексту объявления."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def build_item_text(items: pd.DataFrame):
    """
    Текст объявления для первого lexical baseline.
    title + параметры объявления.
    """
    title = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str)

    return title + " " + params


class BM25Retriever:
    """Классический BM25 по title + item params; используется как global baseline."""

    def __init__(
        self,
        k1=1.5,
        b=0.75,
        min_df=2,
    ):
        self.k1 = k1
        self.b = b

        self.vectorizer = CountVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            lowercase=True,
            dtype=np.float32,
        )

        self.item_matrix = None
        self.item_ids = None

    def fit(self, items: pd.DataFrame):
        """Строит sparse BM25 document matrix и сохраняет соответствующий порядок item_id."""

        item_texts = build_item_text(items)

        # term-frequency matrix: (documents, terms)
        matrix = self.vectorizer.fit_transform(item_texts).tocsr()

        n_docs = matrix.shape[0]

        # Длина каждого документа в токенах
        doc_len = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)

        avg_doc_len = doc_len.mean()

        # В скольких документах встретился каждый term
        df = np.asarray((matrix > 0).sum(axis=0)).ravel()

        # BM25 IDF
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Нормализация длины документа
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_len / avg_doc_len).astype(
            np.float32
        )

        # Для каждого ненулевого элемента надо знать,
        # к какому документу он относится
        row_ids = np.repeat(
            np.arange(n_docs, dtype=np.int32),
            np.diff(matrix.indptr),
        )

        tf = matrix.data

        # BM25 TF component
        matrix.data = (tf * (self.k1 + 1.0) / (tf + length_norm[row_ids])).astype(
            np.float32
        )

        # IDF component
        matrix.data *= idf[matrix.indices]

        del row_ids

        self.item_matrix = matrix
        self.item_ids = items["item_id"].to_numpy()

        return self

    def retrieve(
        self,
        queries,
        query_col="search_query_norm",
        k=50,
        batch_size=256,
    ):
        """Возвращает top-k BM25 кандидатов для каждой строки queries."""

        query_matrix = self.vectorizer.transform(queries[query_col].fillna("")).tocsr()

        query_matrix.data[:] = 1.0

        predictions = {}

        item_matrix_t = self.item_matrix.T.tocsr()

        for start in range(
            0,
            len(queries),
            batch_size,
        ):
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

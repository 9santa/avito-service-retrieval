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


class CharTfidfRetriever:
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

    def fit(self, items: pd.DataFrame):
        """
        Строит TF-IDF индекс по корпусу объявлений.
        """
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
        """
        Возвращает top-k item_id для каждого запроса.

        queries:
            pd.DataFrame, одна строка на retrieval-query.

        Возвращает:
            dict[query_key -> list[item_id]]

        Индекс DataFrame используется как query_key.
        """
        query_matrix = self.vectorizer.transform(queries[query_col].fillna(""))

        predictions = {}

        item_matrix_t = self.item_matrix.T.tocsr()

        total_batches = (len(queries) + batch_size - 1) / batch_size

        for start in tqdm(
            range(0, len(queries), batch_size),
            total=total_batches,
            desc="Retrieving Candidates",
        ):
            end = min(start + batch_size, len(queries))

            # TF-IDF in sklearn is L2-normalized
            # so dot product = cosine similarity
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

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build_dense_item_text(items: pd.DataFrame):
    title = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str)

    # This is the layout intfloat/multilingual-e5-base wants
    return "passage: " + title + " " + params


class GlobalDenseRetriver:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        batch_size=64,
        max_seq_len=256,
        device="cuda",
    ):
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.model.max_seq_length = max_seq_len

        self.batch_size = batch_size
        self.index = None
        self.item_ids = None

    def fit(self, items: pd.DataFrame):
        self.item_ids = items["item_id"].to_numpy()

        texts = build_dense_item_text(items)

        dim = self.model.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(dim)

        # Не держим embeddings всего корпуса в RAM.
        # Кодируем кусками и сразу добавляем в FAISS.
        chunk_size = 4096

        for start in tqdm(
            range(0, len(items), chunk_size),
            desc="Encoding items",
        ):
            end = min(start + chunk_size, len(items))

            embeddings = self.model.encode(
                texts.iloc[start:end].tolist(),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            self.index.add(embeddings)

        return self

    def retrieve(
        self,
        queries: pd.DataFrame,
        query_col="search_query_norm",
        k=50,
    ):
        texts = "query: " + queries[query_col].fillna("").astype(str)

        query_embeddings = self.model.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        scores, indices = self.index.search(
            query_embeddings,
            k=k,
        )

        predictions = {}

        for i, query_key in enumerate(queries.index):
            valid = indices[i] >= 0

            predictions[query_key] = self.item_ids[indices[i][valid]].tolist()

        return predictions

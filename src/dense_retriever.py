from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.local_dense_retriever import build_dense_query_text


def build_dense_item_text(items: pd.DataFrame, prefix="passage: "):
    title = items["item_title_raw"].fillna("").astype(str)
    params = items["item_infm_params_text"].fillna("").astype(str)

    # This is the layout intfloat/multilingual-e5-base wants
    return prefix + title + " " + params


class GlobalDenseRetriever:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        batch_size=64,
        max_seq_len=256,
        device="cuda",
        fp16=False,
        item_prefix="passage: ",
        query_prefix="query: ",
    ):
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.model.max_seq_length = max_seq_len

        if fp16 and device.startswith("cuda"):
            self.model.half()

        self.batch_size = batch_size
        self.item_prefix = item_prefix
        self.query_prefix = query_prefix
        self.index = None
        self.item_ids = None

    def fit(
        self,
        items: pd.DataFrame,
        embeddings_path=None,
        embedding_ids_path=None,
        index_add_batch_size=20_000,
    ):
        self.item_ids = items["item_id"].to_numpy()

        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)
            item_embeddings = np.load(embeddings_path, mmap_mode="r")

            if len(item_embeddings) != len(items):
                raise ValueError("Embedding count != corpus size")

            if embedding_ids_path is None:
                embedding_ids_path = embeddings_path.with_name(
                    embeddings_path.stem + "_item_ids.npy"
                )

            embedding_ids_path = Path(embedding_ids_path)
            if embedding_ids_path.exists():
                cached_ids = np.load(embedding_ids_path, allow_pickle=True)
                if not np.array_equal(cached_ids, self.item_ids):
                    raise ValueError("Item order does not match embedding cache.")

                print("Item ID alignment: OK")

            print("Item embeddings:", item_embeddings.shape)
            self.index = faiss.IndexFlatIP(item_embeddings.shape[1])

            for start in tqdm(
                range(0, len(item_embeddings), index_add_batch_size),
                desc="Building FAISS index",
            ):
                chunk = np.ascontiguousarray(
                    item_embeddings[start : start + index_add_batch_size],
                    dtype=np.float32,
                )
                self.index.add(chunk)

            return self

        texts = build_dense_item_text(items, prefix=self.item_prefix)

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
        return_scores=False,
    ):
        if self.index is None:
            raise RuntimeError("Call fit() before retrieve().")

        if queries.empty:
            return ({}, {}) if return_scores else {}

        if "search_infm_params_text" in queries:
            texts = build_dense_query_text(
                queries.assign(search_query_norm=queries[query_col]),
                prefix=self.query_prefix,
            )
        else:
            texts = self.query_prefix + queries[query_col].fillna("").astype(str)

        codes, unique_texts = pd.factorize(texts, sort=False)
        print("Unique semantic queries:", len(unique_texts))

        query_embeddings = self.model.encode(
            unique_texts.tolist(),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        scores, indices = self.index.search(
            np.ascontiguousarray(query_embeddings),
            k=k,
        )

        predictions = {}
        prediction_scores = {}

        for i, query_key in enumerate(queries.index):
            semantic_idx = codes[i]
            valid = indices[semantic_idx] >= 0

            predictions[query_key] = self.item_ids[indices[semantic_idx][valid]].tolist()
            prediction_scores[query_key] = scores[semantic_idx][valid].astype(float).tolist()

        if return_scores:
            return predictions, prediction_scores
        return predictions

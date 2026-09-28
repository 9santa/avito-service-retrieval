"""Nearest-query microcategory prior, переносимый с исторических кликов на cold queries."""

from collections import defaultdict

import re
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def normalize_text(value) -> str:
    """Нормализует query/params перед дедупликацией semantic queries."""

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value).lower().replace("ё", "е")

    text = re.sub(r"[^0-9a-zа-я]+", " ", text)

    return " ".join(text.split())


def build_semantic_text(df: pd.DataFrame):
    """
    Semantic unit для nearest-query поиска:
        search_query_norm + search_infm_params_text

    Location намеренно не включаем, хотим переносить
    intent/microcat между локациями.
    """
    queries = df["search_query_norm"].fillna("").astype(str).map(normalize_text)

    params = df["search_infm_params_text"].fillna("").astype(str).map(normalize_text)

    result = []

    for q, p in zip(queries, params):
        if p:
            result.append(f"{q} {p}")
        else:
            result.append(q)

    return pd.Series(result, index=df.index)


class MicrocatPrior:
    """Переносит распределение clicked microcats от ближайших historical queries."""

    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        device="cuda",
        encode_batch_size=256,
        max_seq_length=128,
        neighbor_k=30,
        top_microcats=20,
        similarity_power=4.0,
    ):
        self.model_name = model_name
        self.device = device

        self.encode_batch_size = encode_batch_size
        self.max_seq_length = max_seq_length

        self.neighbor_k = neighbor_k
        self.top_microcats = top_microcats
        self.similarity_power = similarity_power

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )
        self.model.max_seq_length = max_seq_length

        if device.startswith("cuda"):
            self.model.half()

        self.history_texts = None
        self.history_microcats = None
        self.history_probs = None

        self.index = None

    # -----------------------------------------------------------------
    # Historical (train_part) query -> microcat distributions
    # -----------------------------------------------------------------

    def fit(self, train_part: pd.DataFrame):
        """Строит P(microcat | historical query) и exact E5 index по historical queries."""

        required = [
            "search_query_norm",
            "search_infm_params_text",
            "item_id",
            "item_microcat_id",
        ]

        missing = [col for col in required if col not in train_part.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        history = train_part[required].copy()

        history = history.dropna(
            subset=[
                "item_microcat_id",
            ]
        )

        history["semantic_text"] = build_semantic_text(history)

        # Одинаковый query-item не должен несколько раз
        # усиливать microcat только из-за дублей interaction rows
        history = history.drop_duplicates(
            subset=[
                "semantic_text",
                "item_id",
            ]
        )

        history["item_microcat_id"] = history["item_microcat_id"].astype(np.int64)

        print("Historical positive pairs:", len(history))
        print("Unique historical semantic queries:", history["semantic_text"].nunique())

        # -------------------------------------------------------------
        # P(microcat | historical semantic query) распределение
        # -------------------------------------------------------------

        counts = (
            history.groupby(
                [
                    "semantic_text",
                    "item_microcat_id",
                ],
                sort=False,
            )
            .size()
            .rename("count")
            .reset_index()
        )

        totals = counts.groupby("semantic_text")["count"].transform("sum")

        counts["prob"] = counts["count"] / totals

        history_texts = []
        history_microcats = []
        history_probs = []

        for semantic_text, group in tqdm(
            counts.groupby("semantic_text", sort=False),
            desc="Building history distributions",
        ):
            history_texts.append(semantic_text)

            history_microcats.append(group["item_microcat_id"].to_numpy(dtype=np.int32))

            history_probs.append(group["prob"].to_numpy(dtype=np.float32))

        self.history_texts = history_texts
        self.history_microcats = history_microcats
        self.history_probs = history_probs

        # -------------------------------------------------------------
        # Encode historical queries
        # -------------------------------------------------------------

        encoder_texts = ["query: " + text for text in self.history_texts]

        print()
        print("Encoding historical queries...")

        history_embeddings = self.model.encode(
            encoder_texts,
            batch_size=self.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

        print("History embeddings:", history_embeddings.shape)

        # -------------------------------------------------------------
        # Exact FAISS cosine (normalized dot product) index
        # -------------------------------------------------------------

        dim = history_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        self.index.add(history_embeddings)

        print("FAISS index size:", self.index.ntotal)

        del history_embeddings

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    # -----------------------------------------------------------------
    # Predict priors
    # -----------------------------------------------------------------

    def predict(self, queries: pd.DataFrame):
        """Агрегирует microcat priors от top-N semantic neighbors для каждой query signature."""

        if self.index is None:
            raise RuntimeError("Index isn't ready. Call fit() first")

        query_text = build_semantic_text(queries)

        # Многие query signatures имеют одинаковый semantic query.
        # Кодируем только unique
        codes, unique_texts = pd.factorize(query_text, sort=False)

        unique_texts = unique_texts.tolist()

        print("Query signatures:", len(queries))
        print("Unique semantic queries:", len(unique_texts))

        encoder_texts = ["query: " + text for text in unique_texts]

        query_embeddings = self.model.encode(
            encoder_texts,
            batch_size=self.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

        k = min(self.neighbor_k, self.index.ntotal)

        print()
        print(f"Searching top-{k} similar historical queries...")

        similarities, indices = self.index.search(query_embeddings, k)

        # -------------------------------------------------------------
        # Build prior once for each unique semantic query.
        # -------------------------------------------------------------

        unique_priors = []

        for query_pos in tqdm(
            range(len(unique_texts)), desc="Aggregating microcat priors"
        ):
            microcat_scores = defaultdict(float)

            microcat_support = defaultdict(int)

            sims = similarities[query_pos]

            neighbor_indices = indices[query_pos]

            total_weight = 0.0

            for sim, neighbor_idx in zip(sims, neighbor_indices):
                if neighbor_idx < 0:
                    continue

                positive_sim = max(float(sim), 0.0)

                weight = positive_sim**self.similarity_power

                if weight <= 0:
                    continue

                total_weight += weight

                microcats = self.history_microcats[neighbor_idx]

                probs = self.history_probs[neighbor_idx]

                for microcat, prob in zip(microcats, probs):
                    microcat = int(microcat)

                    microcat_scores[microcat] += weight * float(prob)

                    microcat_support[microcat] += 1

            if total_weight > 0:
                for microcat in microcat_scores:
                    microcat_scores[microcat] /= total_weight

            ranked = sorted(
                microcat_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            ranked = ranked[: self.top_microcats]

            top_microcats = [int(microcat) for microcat, _ in ranked]

            top_scores = [float(score) for _, score in ranked]

            top_supports = [
                int(microcat_support[microcat]) for microcat in top_microcats
            ]

            valid_sims = [float(x) for x in sims if x > -1]

            top1_sim = valid_sims[0] if valid_sims else 0.0

            mean5_sim = float(np.mean(valid_sims[:5])) if valid_sims else 0.0

            unique_priors.append(
                {
                    "microcats": top_microcats,
                    "scores": top_scores,
                    "supports": top_supports,
                    "neighbor_sim_top1": top1_sim,
                    "neighbor_sim_mean5": mean5_sim,
                }
            )

        # -------------------------------------------------------------
        # Expand back to query_signature.
        # -------------------------------------------------------------

        query_ids = queries.index.to_numpy()

        result = {}

        for row_pos, query_id in enumerate(query_ids):
            result[query_id] = unique_priors[codes[row_pos]]

        return result

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build_dense_item_text(items: pd.DataFrame, prefix="passage: ") -> pd.Series:
    """
    Формирует текст объявления для encoder модели.
    """
    title = items["item_title_raw"].fillna("").astype(str)

    params = items["item_infm_params_text"].fillna("").astype(str)

    return prefix + title + " " + params


def build_dense_query_text(queries: pd.DataFrame, prefix="query: ") -> pd.Series:
    """
    Формирует текст поискового запроса.

    Используем: search_query_norm + search_infm_params_text

    Location намеренно не передается энкодеру:
    он используется как routing-признак.
    """
    query = queries["search_query_norm"].fillna("").astype(str)

    params = queries["search_infm_params_text"].fillna("").astype(str)

    return prefix + query + " " + params


class LocalDenseRetriever:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        device="cuda",
        encode_batch_size=128,
        query_batch_size=128,
        max_seq_length=128,
        fp16=True,
        item_prefix="passage: ",
        query_prefix="query: ",
    ):
        self.model_name = model_name
        self.device = device
        self.item_prefix = item_prefix
        self.query_prefix = query_prefix

        self.encode_batch_size = encode_batch_size
        self.query_batch_size = query_batch_size

        # Загружаем encoder.
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.model.max_seq_length = max_seq_length

        if fp16 and device.startswith("cuda"):
            self.model.half()

        # Будут заполнены после fit().
        self.item_embeddings = None
        self.item_ids = None
        self.location_to_indices = None

    # ============================================================
    # FIT
    # ============================================================

    def fit(
        self,
        items: pd.DataFrame,
        embeddings_path=None,
        force_recompute=False,
    ):
        """
        1. Создаёт item embeddings.
        2. Хранит их полностью в RAM.
        3. При необходимости сохраняет embeddings на диск.
        4. Строит mapping location -> item rows.
        """

        items = items.reset_index(drop=True)

        self.item_ids = items["item_id"].to_numpy()

        # --------------------------------------------------------
        # Location routing
        # --------------------------------------------------------

        self.location_to_indices = {
            location: group.index.to_numpy(dtype=np.int32)
            for location, group in items.groupby(
                "item_location_id",
                dropna=True,
            )
        }

        # --------------------------------------------------------
        # Попытка загрузить готовые embeddings
        # --------------------------------------------------------

        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)

            ids_path = embeddings_path.with_name(embeddings_path.stem + "_item_ids.npy")

            cache_exists = embeddings_path.exists() and ids_path.exists()

            if cache_exists and not force_recompute:
                cached_ids = np.load(
                    ids_path,
                    allow_pickle=True,
                )

                same_ids = np.array_equal(
                    cached_ids,
                    self.item_ids,
                )

                if same_ids:
                    print(f"Loading embeddings: {embeddings_path}")

                    # Обычный np.load:
                    # весь массив загружается в RAM.
                    self.item_embeddings = np.load(embeddings_path)

                    print(
                        "Item embeddings:",
                        self.item_embeddings.shape,
                    )

                    return self

                print("Cached item IDs do not match. Recomputing embeddings.")

        # --------------------------------------------------------
        # Encode всех items
        # --------------------------------------------------------

        print("Encoding items...")

        texts = build_dense_item_text(items, prefix=self.item_prefix)

        self.item_embeddings = self.model.encode(
            texts.tolist(),
            batch_size=self.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(
            np.float32,
            copy=False,
        )

        print(
            "Item embeddings:",
            self.item_embeddings.shape,
        )

        # --------------------------------------------------------
        # Optional cache на диск
        # --------------------------------------------------------

        if embeddings_path is not None:
            embeddings_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                embeddings_path,
                self.item_embeddings,
            )

            np.save(
                ids_path,
                self.item_ids,
                allow_pickle=True,
            )

            print(f"Saved embeddings: {embeddings_path}")

        return self

    # ============================================================
    # QUERY ENCODING
    # ============================================================

    def _encode_queries(
        self,
        queries: pd.DataFrame,
    ) -> np.ndarray:
        """
        Кодируем только уникальные semantic queries.
        """

        texts = build_dense_query_text(queries, prefix=self.query_prefix)

        # Например:
        #
        # texts:
        # iphone
        # samsung
        # iphone
        #
        # codes:
        # [0, 1, 0]
        #
        # unique_texts:
        # [iphone, samsung]

        codes, unique_texts = pd.factorize(
            texts,
            sort=False,
        )

        print("Queries:", len(queries))
        print("Unique queries:", len(unique_texts))

        unique_embeddings = self.model.encode(
            unique_texts.tolist(),
            batch_size=self.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(
            np.float32,
            copy=False,
        )

        # Возвращаем embedding для каждой исходной query row.
        return unique_embeddings[codes]

    # ============================================================
    # EXACT TOP-K
    # ============================================================

    @staticmethod
    def _top_k(
        query_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        k: int,
    ):
        """
        Exact cosine top-k.

        Embeddings нормализованы, поэтому:

            cosine_similarity(q, x)
                =
            q @ x
        """

        n_items = len(item_embeddings)

        if n_items == 0:
            n_queries = len(query_embeddings)

            return (
                np.empty(
                    (n_queries, 0),
                    dtype=np.int32,
                ),
                np.empty(
                    (n_queries, 0),
                    dtype=np.float32,
                ),
            )

        k = min(k, n_items)

        # --------------------------------------------------------
        # Главное вычисление всего dense retrieval.
        #
        # (queries, embedding_dim)
        #           @
        # (embedding_dim, items)
        #
        # ->
        #
        # (queries, items)
        # --------------------------------------------------------

        scores = query_embeddings @ item_embeddings.T

        # --------------------------------------------------------
        # Выбираем top-k
        # --------------------------------------------------------

        if n_items <= k:
            top_positions = np.argsort(
                scores,
                axis=1,
            )[:, ::-1]

        else:
            # Получаем k лучших, пока без порядка.
            top_positions = np.argpartition(
                scores,
                -k,
                axis=1,
            )[:, -k:]

            top_scores = np.take_along_axis(
                scores,
                top_positions,
                axis=1,
            )

            # Теперь сортируем только найденные k.
            order = np.argsort(
                top_scores,
                axis=1,
            )[:, ::-1]

            top_positions = np.take_along_axis(
                top_positions,
                order,
                axis=1,
            )

        top_scores = np.take_along_axis(
            scores,
            top_positions,
            axis=1,
        )

        return top_positions, top_scores

    # ============================================================
    # RETRIEVE
    # ============================================================

    def retrieve(
        self,
        queries: pd.DataFrame,
        k=50,
    ):
        """
        Для каждого query:

        1. encode query
        2. определяем location
        3. берём items только этой location
        4. считаем cosine similarity
        5. возвращаем top-k
        """

        if self.item_embeddings is None:
            raise RuntimeError("Call fit() before retrieve().")

        # --------------------------------------------------------
        # Encode всех queries один раз
        # --------------------------------------------------------

        query_embeddings = self._encode_queries(queries)

        query_keys = queries.index.to_numpy()

        locations = queries["search_location_id"].to_numpy()

        predictions = {}
        prediction_scores = {}

        # --------------------------------------------------------
        # Группируем queries по location
        # --------------------------------------------------------

        location_to_query_positions = {}

        for query_pos, location in enumerate(locations):
            query_key = query_keys[query_pos]

            if pd.isna(location):
                predictions[query_key] = []
                prediction_scores[query_key] = []
                continue

            location_to_query_positions.setdefault(
                location,
                [],
            ).append(query_pos)

        # --------------------------------------------------------
        # Search location-by-location
        # --------------------------------------------------------

        for location, query_positions in tqdm(
            location_to_query_positions.items(),
            desc="Dense retrieval",
        ):
            item_indices = self.location_to_indices.get(location)

            # Нет items в этой location.
            if item_indices is None:
                for query_pos in query_positions:
                    query_key = query_keys[query_pos]

                    predictions[query_key] = []
                    prediction_scores[query_key] = []

                continue

            # ----------------------------------------------------
            # Берём embeddings только items этой location.
            # ----------------------------------------------------

            local_item_embeddings = self.item_embeddings[item_indices]

            query_positions = np.asarray(
                query_positions,
                dtype=np.int32,
            )

            # ----------------------------------------------------
            # Query batching
            # ----------------------------------------------------

            for start in range(
                0,
                len(query_positions),
                self.query_batch_size,
            ):
                end = min(
                    start + self.query_batch_size,
                    len(query_positions),
                )

                positions = query_positions[start:end]

                q_embeddings = query_embeddings[positions]

                # -----------------------------------------------
                # Exact search
                # -----------------------------------------------

                local_top_positions, top_scores = self._top_k(
                    query_embeddings=q_embeddings,
                    item_embeddings=local_item_embeddings,
                    k=k,
                )

                # local_top_positions:
                #
                # позиции внутри local_item_embeddings
                #
                # Нам нужны глобальные rows items.

                global_item_rows = item_indices[local_top_positions]

                # -----------------------------------------------
                # Сохраняем результат
                # -----------------------------------------------

                for local_query_pos, global_query_pos in enumerate(positions):
                    query_key = query_keys[global_query_pos]

                    rows = global_item_rows[local_query_pos]

                    scores = top_scores[local_query_pos]

                    predictions[query_key] = self.item_ids[rows].tolist()

                    prediction_scores[query_key] = scores.astype(float).tolist()

        return predictions, prediction_scores

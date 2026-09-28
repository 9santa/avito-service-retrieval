"""Построение query-item признаков для выбора top-50 из union нескольких retrieval-источников."""

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------


def normalize_text(value) -> str:
    """Лёгкая нормализация для pairwise lexical overlap features."""

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value).lower().replace("ё", "е")

    text = re.sub(r"[^0-9a-zа-я]+", " ", text)

    return " ".join(text.split())


def make_tokens(text: str):
    """
    Splits text into tokens on whitespaces.
    """
    if not text:
        return frozenset()

    return frozenset(text.split())


def overlap_stats(a, b):
    """
    Returns:
        overlap_count
        recall wrt a
        precision wrt b
        jaccard
    """
    if not a or not b:
        return 0.0, 0.0, 0.0, 0.0

    overlap = len(a & b)

    recall = overlap / len(a)
    precision = overlap / len(b)

    union_size = len(a | b)

    jaccard = overlap / union_size if union_size else 0.0

    return (
        float(overlap),
        float(recall),
        float(precision),
        float(jaccard),
    )


# ---------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------

SCORE_SOURCE_NAMES = [
    "char_local",
    "word_local",
    "dense_local",
    "dense_global",
    "dense_microcat",
    "dense_desc_local",
    "dense_desc_global",
]

SCORE_FEATURE_NAMES = [
    f"{source}_{suffix}"
    for source in SCORE_SOURCE_NAMES
    for suffix in [
        "score",
        "gap_to_top",
        "margin_over_last",
        "query_score_range",
    ]
]

FEATURE_NAMES = [
    # source agreement
    "num_sources",
    "local_source_count",
    "global_dense_only",
    "best_present_rank",
    "mean_present_rank",
    "rrf_score",
    # raw retrieval confidence
    *SCORE_FEATURE_NAMES,
    # location
    "same_location",
    "query_location_items_log1p",
    "item_location_items_log1p",
    "query_location_lt50",
    # query <-> title
    "query_title_overlap",
    "query_title_recall",
    "query_title_precision",
    "query_title_jaccard",
    "all_query_tokens_in_title",
    "exact_query_in_title",
    # search params <-> item params
    "qparams_itemparams_overlap",
    "qparams_itemparams_recall",
    "qparams_itemparams_jaccard",
    # search params <-> title
    "qparams_title_recall",
    # item metadata
    "log_price",
    "item_rating",
    "log_reviews",
    "item_phone_hidden",
    "item_message_forbidden",
    "title_len_words",
    "item_params_len_words",
    # microcat prior
    "microcat_prior_score",
    "microcat_prior_rank",
    "microcat_top1_match",
    "microcat_top3_match",
    "microcat_top5_match",
    "microcat_support",
    "dense_microcat_present",
    "neighbor_query_sim_top1",
    "neighbor_query_sim_mean5",
    # dense item description
    "dense_desc_local_rank",
    "dense_desc_global_rank",
    "dense_desc_local_present",
    "dense_desc_global_present",
]


class RankerFeatureBuilder:
    """
    Строит признаки для union семи retrieval-источников.

    Важное:
    - candidate pool НЕ ограничиваем top-50;
    - train groups без positive внутри pool можно отбросить,
      потому что LambdaRank ничего из них не выучит;
    - eval groups без positive не отбрасываем.
    """

    SOURCE_NAMES = [
        "char_local",
        "word_local",
        "dense_local",
        "dense_global",
        "dense_microcat",
        "dense_desc_local",
        "dense_desc_global",
    ]

    LOCAL_SOURCE_NAMES = [
        "char_local",
        "word_local",
        "dense_local",
        "dense_desc_local",
    ]

    def __init__(
        self,
        items: pd.DataFrame,
        queries: pd.DataFrame,
        predictions: dict,
        retrieval_scores: dict,
        microcat_priors: dict,
        missing_rank=51,
        rrf_k=60,
    ):
        self.items = items.reset_index(drop=True)

        self.queries = queries

        self.predictions = predictions
        self.retrieval_scores = retrieval_scores

        for source in SCORE_SOURCE_NAMES:
            if source not in self.retrieval_scores:
                raise ValueError(f"Missing retrieval scores for source: {source}")

        # -------------------------------------------------------------
        # microcats
        # -------------------------------------------------------------
        self.microcat_priors = microcat_priors
        self.item_microcats = self._numeric_column("item_microcat_id", fill_value=-1)

        self.missing_rank = missing_rank
        self.rrf_k = rrf_k

        # -------------------------------------------------------------
        # Item lookup
        # -------------------------------------------------------------

        self.item_ids = self.items["item_id"].to_numpy()

        self.item_row_by_id = {item_id: i for i, item_id in enumerate(self.item_ids)}

        self.item_titles = (
            self.items["item_title_raw"].fillna("").astype(str).to_numpy()
        )

        self.item_params = (
            self.items["item_infm_params_text"].fillna("").astype(str).to_numpy()
        )

        self.item_locations = self.items["item_location_id"].to_numpy()

        # -------------------------------------------------------------
        # Numeric item metadata
        # -------------------------------------------------------------

        self.item_price = self._numeric_column("item_price")

        self.item_rating = self._numeric_column("item_rating")

        self.item_reviews = self._numeric_column("item_rating_reviews_count")

        self.item_phone_hidden = self._numeric_column("item_is_phone_hidden")

        self.item_message_forbidden = self._numeric_column("item_is_message_forbidden")

        # -------------------------------------------------------------
        # Corpus location sizes
        # -------------------------------------------------------------

        location_counts = (
            self.items["item_location_id"].value_counts(dropna=True).to_dict()
        )

        self.location_counts = location_counts

        # -------------------------------------------------------------
        # Query cache
        # -------------------------------------------------------------

        self.query_info = {}

        for query_id, row in tqdm(
            queries.iterrows(),
            total=len(queries),
            desc="Preparing queries",
        ):
            q = normalize_text(row["search_query_norm"])

            qparams = normalize_text(row["search_infm_params_text"])

            self.query_info[query_id] = {
                "query_norm": q,
                "query_tokens": make_tokens(q),
                "params_norm": qparams,
                "params_tokens": make_tokens(qparams),
                "location": row["search_location_id"],
            }

    def _numeric_column(self, name, *, fill_value=0):
        """
        Преобразует item metadata в numeric ndarray с единым fill value.
        """
        return (
            pd.to_numeric(
                self.items[name],
                errors="coerce",
            )
            .fillna(fill_value)
            .to_numpy(dtype=np.float32)
        )

    # -----------------------------------------------------------------
    # Item text cache
    #
    # Ограниченный cache, чтобы не раздувать RAM.
    # -----------------------------------------------------------------

    @lru_cache(maxsize=50_000)
    def _item_text_info(
        self,
        row_idx,
    ):
        """Кэширует нормализованный item text и tokens для повторных pair features."""

        title = normalize_text(self.item_titles[row_idx])

        params = normalize_text(self.item_params[row_idx])

        title_tokens = make_tokens(title)
        params_tokens = make_tokens(params)

        return (
            title,
            params,
            title_tokens,
            params_tokens,
            len(title_tokens),
            len(params_tokens),
        )

    # -----------------------------------------------------------------
    # Candidate generation context
    # -----------------------------------------------------------------

    def candidate_union(
        self,
        query_id,
    ):
        """
        Ordered union всех retrieval-источников без ограничения k=50.
        """
        # Так как set не сохраняет порядок, используем список
        result = []
        seen = set()

        for source_name in self.SOURCE_NAMES:
            source = self.predictions[source_name]

            for item_id in source.get(
                query_id,
                [],
            ):
                # На всякий случай исключаем item,
                # которого нет в corpus.
                if item_id not in self.item_row_by_id:
                    continue

                if item_id in seen:
                    continue

                seen.add(item_id)
                result.append(item_id)

        return result

    def _source_context(
        self,
        query_id,
    ):
        """
        Возвращает:
            candidates
            rank maps
            per-source score maps and query-level score stats
        """
        rank_maps = {}

        candidates = []
        seen = set()

        for source_name in self.SOURCE_NAMES:
            source_items = self.predictions[source_name].get(
                query_id,
                [],
            )

            rank_map = {}

            for rank, item_id in enumerate(
                source_items,
                start=1,
            ):
                if item_id not in self.item_row_by_id:
                    continue

                rank_map[item_id] = rank

                if item_id not in seen:
                    seen.add(item_id)
                    candidates.append(item_id)

            rank_maps[source_name] = rank_map

        score_contexts = {}

        for source_name in SCORE_SOURCE_NAMES:
            source_items = self.predictions[source_name].get(query_id, [])
            source_scores = self.retrieval_scores[source_name].get(query_id, [])

            if len(source_items) != len(source_scores):
                raise ValueError(
                    f"Prediction/score length mismatch "
                    f"for {source_name}, query={query_id}: "
                    f"{len(source_items)} items vs {len(source_scores)} scores"
                )

            score_map = {
                item_id: float(score)
                for item_id, score in zip(source_items, source_scores)
            }

            if len(source_scores):
                top_score = float(source_scores[0])
                last_score = float(source_scores[-1])
            else:
                top_score = 0.0
                last_score = 0.0

            score_contexts[source_name] = {
                "score_map": score_map,
                "top_score": top_score,
                "last_score": last_score,
                "query_range": top_score - last_score,
            }

        return (
            candidates,
            rank_maps,
            score_contexts,
        )

    # -----------------------------------------------------------------
    # One query group
    # -----------------------------------------------------------------

    def build_group(
        self,
        query_id,
    ):
        """Строит candidate list и feature matrix для одной query group."""

        (
            candidates,
            rank_maps,
            score_contexts,
        ) = self._source_context(query_id)

        n = len(candidates)

        X = np.zeros(
            (
                n,
                len(FEATURE_NAMES),
            ),
            dtype=np.float32,
        )

        qinfo = self.query_info[query_id]

        query_norm = qinfo["query_norm"]

        query_tokens = qinfo["query_tokens"]

        qparams_tokens = qinfo["params_tokens"]

        query_location = qinfo["location"]

        if pd.isna(query_location):
            query_location_count = 0
        else:
            query_location_count = self.location_counts.get(
                query_location,
                0,
            )

        query_location_log = np.log1p(query_location_count)

        query_location_lt50 = float(query_location_count < 50)

        # -----------------------------------------------------------------
        # Microcat prior features
        # -----------------------------------------------------------------

        prior = self.microcat_priors.get(
            query_id,
            {
                "microcats": [],
                "scores": [],
                "supports": [],
                "neighbor_sim_top1": 0.0,
                "neighbor_sim_mean5": 0.0,
            },
        )

        prior_microcats = prior["microcats"]
        prior_scores = prior["scores"]
        prior_supports = prior["supports"]

        microcat_score_map = {
            microcat: score
            for microcat, score in zip(
                prior_microcats,
                prior_scores,
            )
        }

        microcat_rank_map = {
            microcat: rank
            for rank, microcat in enumerate(
                prior_microcats,
                start=1,
            )
        }

        microcat_support_map = {
            microcat: support
            for microcat, support in zip(
                prior_microcats,
                prior_supports,
            )
        }

        neighbor_query_sim_top1 = float(prior["neighbor_sim_top1"])

        neighbor_query_sim_mean5 = float(prior["neighbor_sim_mean5"])

        for j, item_id in enumerate(candidates):
            item_row = self.item_row_by_id[item_id]

            # ---------------------------------------------------------
            # Rank / presence
            # ---------------------------------------------------------

            char_rank = rank_maps["char_local"].get(
                item_id,
                self.missing_rank,
            )

            word_rank = rank_maps["word_local"].get(
                item_id,
                self.missing_rank,
            )

            dense_local_rank = rank_maps["dense_local"].get(
                item_id,
                self.missing_rank,
            )

            dense_global_rank = rank_maps["dense_global"].get(
                item_id,
                self.missing_rank,
            )

            dense_microcat_rank = rank_maps["dense_microcat"].get(
                item_id,
                self.missing_rank,
            )

            dense_desc_local_rank = rank_maps["dense_desc_local"].get(
                item_id, self.missing_rank
            )

            dense_desc_global_rank = rank_maps["dense_desc_global"].get(
                item_id, self.missing_rank
            )

            char_present = float(char_rank != self.missing_rank)

            word_present = float(word_rank != self.missing_rank)

            dense_local_present = float(dense_local_rank != self.missing_rank)

            dense_global_present = float(dense_global_rank != self.missing_rank)

            dense_microcat_present = float(dense_microcat_rank != self.missing_rank)

            dense_desc_local_present = float(dense_desc_local_rank != self.missing_rank)

            dense_desc_global_present = float(
                dense_desc_global_rank != self.missing_rank
            )

            present_ranks = []

            for rank in [
                char_rank,
                word_rank,
                dense_local_rank,
                dense_global_rank,
                dense_microcat_rank,
                dense_desc_local_rank,
                dense_desc_global_rank,
            ]:
                if rank != self.missing_rank:
                    present_ranks.append(rank)

            num_sources = len(present_ranks)

            local_source_count = int(
                char_present
                + word_present
                + dense_local_present
                + dense_desc_local_present
            )

            global_dense_only = float(dense_global_present and local_source_count == 0)

            best_present_rank = (
                min(present_ranks) if present_ranks else self.missing_rank
            )

            mean_present_rank = (
                float(np.mean(present_ranks))
                if present_ranks
                else float(self.missing_rank)
            )

            rrf_score = 0.0

            for rank in present_ranks:
                rrf_score += 1.0 / (self.rrf_k + rank)

            # ---------------------------------------------------------
            # Retrieval confidence
            # ---------------------------------------------------------

            score_feature_values = []

            for source_name in SCORE_SOURCE_NAMES:
                context = score_contexts[source_name]
                score_map = context["score_map"]

                if item_id in score_map:
                    raw_score = float(score_map[item_id])
                    gap_to_top = context["top_score"] - raw_score
                    margin_over_last = raw_score - context["last_score"]
                else:
                    raw_score = 0.0
                    gap_to_top = 0.0
                    margin_over_last = 0.0

                query_score_range = context["query_range"]

                score_feature_values.extend(
                    [
                        raw_score,
                        gap_to_top,
                        margin_over_last,
                        query_score_range,
                    ]
                )

            # ---------------------------------------------------------
            # Location
            # ---------------------------------------------------------

            item_location = self.item_locations[item_row]

            if pd.isna(query_location) or pd.isna(item_location):
                same_location = 0.0
            else:
                same_location = float(query_location == item_location)

            if pd.isna(item_location):
                item_location_count = 0
            else:
                item_location_count = self.location_counts.get(
                    item_location,
                    0,
                )

            item_location_log = np.log1p(item_location_count)

            # ---------------------------------------------------------
            # Item text
            # ---------------------------------------------------------

            (
                title_norm,
                _,
                title_tokens,
                item_params_tokens,
                title_len,
                item_params_len,
            ) = self._item_text_info(item_row)

            # query <-> title
            (
                query_title_overlap,
                query_title_recall,
                query_title_precision,
                query_title_jaccard,
            ) = overlap_stats(
                query_tokens,
                title_tokens,
            )

            if query_tokens:
                all_query_tokens_in_title = float(query_tokens.issubset(title_tokens))
            else:
                all_query_tokens_in_title = 0.0

            exact_query_in_title = float(bool(query_norm) and query_norm in title_norm)

            # qparams <-> item params
            (
                qparams_itemparams_overlap,
                qparams_itemparams_recall,
                _,
                qparams_itemparams_jaccard,
            ) = overlap_stats(
                qparams_tokens,
                item_params_tokens,
            )

            # qparams <-> title
            (
                _,
                qparams_title_recall,
                _,
                _,
            ) = overlap_stats(
                qparams_tokens,
                title_tokens,
            )

            # ---------------------------------------------------------
            # Item metadata
            # ---------------------------------------------------------

            price = float(self.item_price[item_row])

            price = max(
                price,
                0.0,
            )

            log_price = np.log1p(price)

            rating = float(self.item_rating[item_row])

            reviews = max(
                float(self.item_reviews[item_row]),
                0.0,
            )

            log_reviews = np.log1p(reviews)

            phone_hidden = float(self.item_phone_hidden[item_row])

            message_forbidden = float(self.item_message_forbidden[item_row])

            # ---------------------------------------------------------
            # Microcat prior
            # ---------------------------------------------------------

            item_microcat = int(self.item_microcats[item_row])

            microcat_prior_score = float(
                microcat_score_map.get(
                    item_microcat,
                    0.0,
                )
            )

            microcat_prior_rank = float(
                microcat_rank_map.get(
                    item_microcat,
                    21,
                )
            )

            microcat_top1_match = float(item_microcat in prior_microcats[:1])

            microcat_top3_match = float(item_microcat in prior_microcats[:3])

            microcat_top5_match = float(item_microcat in prior_microcats[:5])

            microcat_support = float(
                microcat_support_map.get(
                    item_microcat,
                    0,
                )
            )

            # ---------------------------------------------------------
            # Final row
            # ---------------------------------------------------------

            X[j] = [
                # agreement
                num_sources,
                local_source_count,
                global_dense_only,
                best_present_rank,
                mean_present_rank,
                rrf_score,
                # all retrieval scores
                *score_feature_values,
                # location
                same_location,
                query_location_log,
                item_location_log,
                query_location_lt50,
                # query-title
                query_title_overlap,
                query_title_recall,
                query_title_precision,
                query_title_jaccard,
                all_query_tokens_in_title,
                exact_query_in_title,
                # params-params
                qparams_itemparams_overlap,
                qparams_itemparams_recall,
                qparams_itemparams_jaccard,
                # params-title
                qparams_title_recall,
                # metadata
                log_price,
                rating,
                log_reviews,
                phone_hidden,
                message_forbidden,
                title_len,
                item_params_len,
                # microcat prior
                microcat_prior_score,
                microcat_prior_rank,
                microcat_top1_match,
                microcat_top3_match,
                microcat_top5_match,
                microcat_support,
                dense_microcat_present,
                neighbor_query_sim_top1,
                neighbor_query_sim_mean5,
                # dense item description
                dense_desc_local_rank,
                dense_desc_global_rank,
                dense_desc_local_present,
                dense_desc_global_present,
            ]

        return candidates, X

    # -----------------------------------------------------------------
    # Dataset building
    # -----------------------------------------------------------------

    def build_matrix(
        self,
        query_ids,
        labels,
        skip_groups_without_positive=False,
        desc="Building features",
    ):
        """
        Two-pass builder.

        Не сохраняет огромные Python DataFrame.
        Сразу создаёт компактные float32 ndarray.
        """

        kept_query_ids = []
        group_sizes = []

        skipped_no_candidates = 0
        skipped_no_positive = 0

        # -------------------------------------------------------------
        # Pass 1:
        # узнаём точный размер matrix.
        # -------------------------------------------------------------

        for query_id in tqdm(
            query_ids,
            desc=f"{desc}: sizing",
        ):
            candidates = self.candidate_union(query_id)

            if not candidates:
                skipped_no_candidates += 1
                continue

            if skip_groups_without_positive:
                positives = labels.get(
                    query_id,
                    set(),
                )

                has_positive = any(item_id in positives for item_id in candidates)

                if not has_positive:
                    skipped_no_positive += 1
                    continue

            kept_query_ids.append(query_id)

            group_sizes.append(len(candidates))

        total_rows = int(np.sum(group_sizes))

        print()
        print(
            f"{desc}: kept groups:",
            len(kept_query_ids),
        )

        print(
            f"{desc}: rows:",
            total_rows,
        )

        print(
            f"{desc}: skipped no candidates:",
            skipped_no_candidates,
        )

        print(
            f"{desc}: skipped no positive:",
            skipped_no_positive,
        )

        # -------------------------------------------------------------
        # Allocate
        # -------------------------------------------------------------

        X = np.empty(
            (
                total_rows,
                len(FEATURE_NAMES),
            ),
            dtype=np.float32,
        )

        y = np.zeros(
            total_rows,
            dtype=np.int8,
        )

        # -------------------------------------------------------------
        # Pass 2
        # -------------------------------------------------------------

        offset = 0

        for query_id, group_size in tqdm(
            zip(
                kept_query_ids,
                group_sizes,
            ),
            total=len(kept_query_ids),
            desc=desc,
        ):
            (
                candidates,
                group_X,
            ) = self.build_group(query_id)

            end = offset + group_size

            X[offset:end] = group_X

            positives = labels.get(
                query_id,
                set(),
            )

            y[offset:end] = np.fromiter(
                (1 if item_id in positives else 0 for item_id in candidates),
                dtype=np.int8,
                count=group_size,
            )

            offset = end

        return {
            "X": X,
            "y": y,
            "group": np.asarray(
                group_sizes,
                dtype=np.int32,
            ),
            "query_ids": kept_query_ids,
            "skipped_no_candidates": (skipped_no_candidates),
            "skipped_no_positive": (skipped_no_positive),
        }

    # -----------------------------------------------------------------
    # Convert flat model scores back to top-k predictions
    # -----------------------------------------------------------------

    def scores_to_predictions(
        self,
        query_ids,
        group_sizes,
        scores,
        all_query_ids=None,
        k=50,
    ):
        """Преобразует flat model scores обратно в top-k item_id по query group boundaries."""

        if all_query_ids is None:
            predictions = {}
        else:
            predictions = {query_id: [] for query_id in all_query_ids}

        offset = 0

        for query_id, group_size in zip(
            query_ids,
            group_sizes,
        ):
            candidates = self.candidate_union(query_id)

            group_scores = scores[offset : offset + group_size]

            take_k = min(
                k,
                group_size,
            )

            if group_size <= take_k:
                order = np.argsort(group_scores)[::-1]
            else:
                top = np.argpartition(
                    group_scores,
                    -take_k,
                )[-take_k:]

                order = top[np.argsort(group_scores[top])[::-1]]

            predictions[query_id] = [candidates[i] for i in order]

            offset += group_size

        return predictions

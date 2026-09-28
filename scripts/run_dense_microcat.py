import gc
import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


from src.data import (
    load_data,
    build_train_item_corpus,
)
from src.fusion import union_predictions
from src.metrics import build_labels, recall_at_k
from src.microcat_prior import build_semantic_text


# =====================================================================
# Paths
# =====================================================================

DATA_DIR = "data/raw"

VAL_PATH = Path("data/interim/val_part.parquet")

ITEM_EMBEDDINGS_PATH = Path("artifacts/embeddings/e5_base_items.npy")

ITEM_IDS_CACHE_PATH = Path("artifacts/embeddings/e5_base_items_item_ids.npy")

MICROCAT_PRIOR_PATH = Path("artifacts/features/microcat_prior_val.pkl")

PRED_DIR = Path("artifacts/predictions")


# Existing sources
CHAR_LOCAL_PATH = PRED_DIR / "char_local_val.pkl"

WORD_LOCAL_PATH = PRED_DIR / "word_local_val.pkl"

DENSE_LOCAL_PATH = PRED_DIR / "dense_local_val.pkl"

DENSE_GLOBAL_PATH = PRED_DIR / "dense_val.pkl"


# =====================================================================
# Config
# =====================================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

ENCODE_BATCH_SIZE = 128
MAX_SEQ_LENGTH = 128

# Считаем результаты максимум по top-10 microcats.
MAX_MICROCATS = 10

# Внутри каждого microcat достаточно top-50:
# итоговая метрика всё равно Recall@50.
PER_MICROCAT_K = 50

# FAISS search batch внутри одного microcat.
SEARCH_QUERY_BATCH = 1024

CUTOFFS = [
    1,
    3,
    5,
    10,
]


# =====================================================================
# Helpers
# =====================================================================


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def mean_candidates(predictions):
    return np.mean([len(items) for items in predictions.values()])


def select_top_k(
    rows,
    scores,
    item_ids,
    k=50,
):
    """
    Берём top-k по dense score из уже посчитанных
    candidates.

    rows:
        corpus row indices

    scores:
        cosine similarity
    """
    valid = (rows >= 0) & np.isfinite(scores)

    rows = rows[valid]
    scores = scores[valid]

    if len(rows) == 0:
        return [], []

    take_k = min(
        k,
        len(rows),
    )

    if len(rows) > take_k:
        top = np.argpartition(
            scores,
            -take_k,
        )[-take_k:]

        top = top[np.argsort(scores[top])[::-1]]
    else:
        top = np.argsort(scores)[::-1]

    selected_rows = rows[top]

    selected_scores = scores[top].astype(float).tolist()

    selected_items = item_ids[selected_rows].tolist()

    return (
        selected_items,
        selected_scores,
    )


# =====================================================================
# Main
# =====================================================================


def main():
    # -----------------------------------------------------------------
    # Load corpus / validation
    # -----------------------------------------------------------------

    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    train, _, _ = load_data(DATA_DIR)

    val_part = pd.read_parquet(VAL_PATH)

    items = build_train_item_corpus(train).reset_index(drop=True)

    val_queries = (
        val_part[
            [
                "query_signature",
                "search_query_norm",
                "search_infm_params_text",
            ]
        ]
        .drop_duplicates("query_signature")
        .set_index("query_signature")
    )

    labels = build_labels(
        val_part,
        query_col="query_signature",
    )

    item_ids = items["item_id"].to_numpy()

    print(
        "Corpus items:",
        len(items),
    )

    print(
        "Validation signatures:",
        len(val_queries),
    )

    # -----------------------------------------------------------------
    # Load item embeddings
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOAD ITEM EMBEDDINGS")
    print("=" * 80)

    item_embeddings = np.load(
        ITEM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    print(
        "Item embeddings:",
        item_embeddings.shape,
    )

    if len(item_embeddings) != len(items):
        raise ValueError("Item embedding count does not match corpus.")

    # Если cache IDs существует — дополнительно проверяем порядок.
    if ITEM_IDS_CACHE_PATH.exists():
        cached_ids = np.load(
            ITEM_IDS_CACHE_PATH,
            allow_pickle=True,
        )

        if not np.array_equal(
            cached_ids,
            item_ids,
        ):
            raise ValueError("Item IDs do not match embedding cache.")

        print("Item ID alignment: OK")

    # -----------------------------------------------------------------
    # Microcat -> corpus row indices
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILD MICROCAT ROUTING")
    print("=" * 80)

    item_microcats = (
        pd.to_numeric(
            items["item_microcat_id"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )

    routing_df = pd.DataFrame(
        {
            "microcat": item_microcats,
            "row": np.arange(
                len(items),
                dtype=np.int32,
            ),
        }
    )

    routing_df = routing_df[routing_df["microcat"] >= 0]

    microcat_to_rows = {
        int(microcat): group["row"].to_numpy(dtype=np.int32)
        for microcat, group in routing_df.groupby(
            "microcat",
            sort=False,
        )
    }

    del routing_df
    gc.collect()

    print(
        "Corpus microcats:",
        len(microcat_to_rows),
    )

    sizes = np.asarray([len(rows) for rows in microcat_to_rows.values()])

    print(
        "Mean items / microcat:",
        f"{sizes.mean():.2f}",
    )

    print(
        "Median items / microcat:",
        f"{np.median(sizes):.2f}",
    )

    print(
        "Max items / microcat:",
        sizes.max(),
    )

    # -----------------------------------------------------------------
    # Load priors
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOAD MICROCAT PRIORS")
    print("=" * 80)

    microcat_priors = load_pickle(MICROCAT_PRIOR_PATH)

    print(
        "Priors:",
        len(microcat_priors),
    )

    # -----------------------------------------------------------------
    # Deduplicate semantic queries
    #
    # dense_microcat не зависит от location.
    # Поэтому одинаковые query+params можно искать только один раз.
    # -----------------------------------------------------------------

    semantic_text = build_semantic_text(val_queries)

    codes, unique_texts = pd.factorize(
        semantic_text,
        sort=False,
    )

    unique_texts = unique_texts.tolist()

    query_ids = val_queries.index.to_numpy()

    n_unique = len(unique_texts)

    first_position = np.full(
        n_unique,
        -1,
        dtype=np.int32,
    )

    for position, code in enumerate(codes):
        if first_position[code] == -1:
            first_position[code] = position

    unique_query_ids = query_ids[first_position]

    print(
        "Signatures:",
        len(query_ids),
    )

    print(
        "Unique semantic queries:",
        n_unique,
    )

    # -----------------------------------------------------------------
    # Encode queries
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("ENCODE QUERIES")
    print("=" * 80)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    model.half()

    encoder_texts = ["query: " + text for text in unique_texts]

    query_embeddings = model.encode(
        encoder_texts,
        batch_size=(ENCODE_BATCH_SIZE),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "Query embeddings:",
        query_embeddings.shape,
    )

    # Encoder больше не нужен.
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Allocate result cube
    #
    # Для каждого unique query:
    #
    #   [microcat rank 0..9] × [top50 item]
    #
    # Храним corpus row, а не item_id:
    # int32 заметно экономнее.
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("PREPARE MICROCAT REQUESTS")
    print("=" * 80)

    result_rows = np.full(
        (
            n_unique,
            MAX_MICROCATS,
            PER_MICROCAT_K,
        ),
        -1,
        dtype=np.int32,
    )

    result_scores = np.full(
        (
            n_unique,
            MAX_MICROCATS,
            PER_MICROCAT_K,
        ),
        -np.inf,
        dtype=np.float32,
    )

    # microcat ->
    #     list unique query indices
    #     list prior rank slots
    requests = defaultdict(
        lambda: [
            [],
            [],
        ]
    )

    missing_prior = 0
    missing_microcat = 0

    for unique_idx, query_id in enumerate(unique_query_ids):
        prior = microcat_priors.get(query_id)

        if prior is None:
            missing_prior += 1
            continue

        predicted_microcats = prior["microcats"][:MAX_MICROCATS]

        for slot, microcat in enumerate(predicted_microcats):
            microcat = int(microcat)

            if microcat not in (microcat_to_rows):
                missing_microcat += 1
                continue

            requests[microcat][0].append(unique_idx)

            requests[microcat][1].append(slot)

    print(
        "Requested microcats:",
        len(requests),
    )

    print(
        "Unique queries without prior:",
        missing_prior,
    )

    print(
        "Predicted microcats absent from corpus:",
        missing_microcat,
    )

    # -----------------------------------------------------------------
    # Search microcat by microcat
    #
    # Для каждого microcat:
    #   1. берём все corpus embeddings этого microcat
    #   2. создаём маленький exact FAISS index
    #   3. одним batch search обслуживаем ВСЕ queries,
    #      которым этот microcat нужен
    #
    # После обработки index удаляется.
    # Поэтому в памяти нет сотен FAISS indices одновременно.
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("DENSE MICROCAT SEARCH")
    print("=" * 80)

    dim = item_embeddings.shape[1]

    for microcat, (
        requested_queries,
        requested_slots,
    ) in tqdm(
        requests.items(),
        desc="Microcats",
    ):
        corpus_rows = microcat_to_rows[microcat]

        # Materialize только текущий microcat
        # из mmap.
        embeddings = np.asarray(
            item_embeddings[corpus_rows],
            dtype=np.float32,
        )

        embeddings = np.ascontiguousarray(embeddings)

        index = faiss.IndexFlatIP(dim)

        index.add(embeddings)

        requested_queries = np.asarray(
            requested_queries,
            dtype=np.int32,
        )

        requested_slots = np.asarray(
            requested_slots,
            dtype=np.int32,
        )

        search_k = min(
            PER_MICROCAT_K,
            len(corpus_rows),
        )

        for start in range(
            0,
            len(requested_queries),
            SEARCH_QUERY_BATCH,
        ):
            end = min(
                start + SEARCH_QUERY_BATCH,
                len(requested_queries),
            )

            qidx = requested_queries[start:end]

            slots = requested_slots[start:end]

            D, I = index.search(
                np.ascontiguousarray(query_embeddings[qidx]),
                search_k,
            )

            # FAISS I — позиции внутри
            # текущего microcat index.
            mapped_rows = np.full(
                I.shape,
                -1,
                dtype=np.int32,
            )

            valid = I >= 0

            mapped_rows[valid] = corpus_rows[I[valid]]

            # Pairwise advanced indexing:
            # result_rows[qidx[i], slots[i], :]
            result_rows[
                qidx,
                slots,
                :search_k,
            ] = mapped_rows

            result_scores[
                qidx,
                slots,
                :search_k,
            ] = D.astype(
                np.float32,
                copy=False,
            )

        del index
        del embeddings

    gc.collect()

    # -----------------------------------------------------------------
    # Build predictions for top-1 / 3 / 5 / 10 microcats
    #
    # Search уже полностью выполнен.
    # Здесь только выбираем top50 среди:
    #
    # top1 -> <=  50 scored candidates
    # top3 -> <= 150
    # top5 -> <= 250
    # top10 -> <=500
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILD CUTOFF PREDICTIONS")
    print("=" * 80)

    all_predictions = {}
    all_prediction_scores = {}

    for cutoff in CUTOFFS:
        print()
        print(f"Building top-{cutoff} microcat retrieval...")

        unique_predictions = [None] * n_unique

        unique_scores = [None] * n_unique

        for unique_idx in tqdm(
            range(n_unique),
            desc=f"top{cutoff}",
        ):
            rows = result_rows[
                unique_idx,
                :cutoff,
                :,
            ].reshape(-1)

            scores = result_scores[
                unique_idx,
                :cutoff,
                :,
            ].reshape(-1)

            (
                selected_items,
                selected_scores,
            ) = select_top_k(
                rows=rows,
                scores=scores,
                item_ids=item_ids,
                k=50,
            )

            unique_predictions[unique_idx] = selected_items

            unique_scores[unique_idx] = selected_scores

        # Expand back from semantic query
        # to query_signature.
        predictions = {
            query_id: unique_predictions[code]
            for query_id, code in zip(
                query_ids,
                codes,
            )
        }

        prediction_scores = {
            query_id: unique_scores[code]
            for query_id, code in zip(
                query_ids,
                codes,
            )
        }

        all_predictions[cutoff] = predictions

        all_prediction_scores[cutoff] = prediction_scores

    # -----------------------------------------------------------------
    # Existing sources
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOAD CURRENT CANDIDATE SOURCES")
    print("=" * 80)

    char_local = load_pickle(CHAR_LOCAL_PATH)

    word_local = load_pickle(WORD_LOCAL_PATH)

    dense_local = load_pickle(DENSE_LOCAL_PATH)

    dense_global = load_pickle(DENSE_GLOBAL_PATH)

    current_sources = [
        char_local,
        word_local,
        dense_local,
        dense_global,
    ]

    current_pool = union_predictions(current_sources)

    current_pool_recall = recall_at_k(
        labels,
        current_pool,
        k=1000,
    )

    current_pool_sizes = np.asarray([len(x) for x in current_pool.values()])

    print(
        "Current 4-source pool recall:",
        f"{current_pool_recall:.6f}",
    )

    print(
        "Current mean pool size:",
        f"{current_pool_sizes.mean():.2f}",
    )

    del current_pool
    gc.collect()

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("DENSE MICROCAT RESULTS")
    print("=" * 80)

    results = []

    for cutoff in CUTOFFS:
        predictions = all_predictions[cutoff]

        print()
        print(f"Top-{cutoff} predicted microcats")

        standalone_scores = {}

        for k in [
            10,
            20,
            50,
        ]:
            score = recall_at_k(
                labels,
                predictions,
                k=k,
            )

            standalone_scores[k] = score

            print(f"  Recall@{k}: {score:.6f}")

        print(
            "  Mean candidates:",
            f"{mean_candidates(predictions):.2f}",
        )

        # -------------------------------------------------------------
        # Add to existing 4-source pool
        # -------------------------------------------------------------

        expanded_pool = union_predictions(current_sources + [predictions])

        expanded_recall = recall_at_k(
            labels,
            expanded_pool,
            k=1000,
        )

        pool_sizes = np.asarray([len(x) for x in expanded_pool.values()])

        gain = expanded_recall - current_pool_recall

        print(
            "  5-source pool recall:",
            f"{expanded_recall:.6f}",
        )

        print(
            "  Pool gain:",
            f"{gain:+.6f}",
        )

        print(
            "  Mean pool size:",
            f"{pool_sizes.mean():.2f}",
        )

        results.append(
            {
                "top_microcats": cutoff,
                "recall@10": standalone_scores[10],
                "recall@20": standalone_scores[20],
                "recall@50": standalone_scores[50],
                "pool_recall": expanded_recall,
                "pool_gain": gain,
                "mean_pool": pool_sizes.mean(),
            }
        )

        del expanded_pool

    # -----------------------------------------------------------------
    # Save predictions + scores
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVE")
    print("=" * 80)

    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for cutoff in CUTOFFS:
        pred_path = PRED_DIR / (f"dense_microcat_top{cutoff}_val.pkl")

        score_path = PRED_DIR / (f"dense_microcat_top{cutoff}_scores_val.pkl")

        with open(
            pred_path,
            "wb",
        ) as f:
            pickle.dump(
                all_predictions[cutoff],
                f,
                protocol=(pickle.HIGHEST_PROTOCOL),
            )

        with open(
            score_path,
            "wb",
        ) as f:
            pickle.dump(
                all_prediction_scores[cutoff],
                f,
                protocol=(pickle.HIGHEST_PROTOCOL),
            )

        print(
            "Saved:",
            pred_path,
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    result_df = pd.DataFrame(results)

    print(
        result_df.to_string(
            index=False,
            float_format=(lambda x: f"{x:.6f}"),
        )
    )


if __name__ == "__main__":
    main()

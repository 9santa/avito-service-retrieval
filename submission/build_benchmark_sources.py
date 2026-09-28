import gc
import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data import load_data
from src.validation import normalize_query
from src.local_retrieval import (
    LocalWordTfidfRetriever,
    LocalCharTfidfRetriever,
)
from src.local_dense_retriever import (
    LocalDenseRetriever,
    build_dense_query_text,
)
from src.microcat_prior import MicrocatPrior


DATA_DIR = "data/raw"

PRED_DIR = Path("artifacts/predictions")
FEATURE_DIR = Path("artifacts/features")
EMBED_DIR = Path("artifacts/embeddings")

MODEL_NAME = "intfloat/multilingual-e5-base"

K = 50
ENCODE_BATCH_SIZE = 64


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(
            obj,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def build_desc_text(items):
    title = items["item_title_raw"].fillna("").astype(str)

    params = items["item_infm_params_text"].fillna("").astype(str)

    desc = items["item_description_raw"].fillna("").astype(str)

    return "passage: " + title + " " + params + " " + desc


def build_desc_embeddings(
    model,
    items,
    path,
):
    if path.exists():
        print("Loading:", path)

        return np.load(
            path,
            mmap_mode="r",
        )

    texts = build_desc_text(items)

    dim = model.get_sentence_embedding_dimension()

    mmap = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(items),
            dim,
        ),
    )

    chunk_size = 2048

    for start in tqdm(
        range(
            0,
            len(items),
            chunk_size,
        ),
        desc="Description embeddings",
    ):
        end = min(
            start + chunk_size,
            len(items),
        )

        emb = model.encode(
            texts.iloc[start:end].tolist(),
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(
            np.float32,
            copy=False,
        )

        mmap[start:end] = emb

    mmap.flush()
    del mmap

    return np.load(
        path,
        mmap_mode="r",
    )


def global_dense_search(
    queries,
    query_embeddings,
    item_embeddings,
    item_ids,
):
    dim = item_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    for start in tqdm(
        range(
            0,
            len(item_embeddings),
            20_000,
        ),
        desc="FAISS add",
    ):
        end = min(
            start + 20_000,
            len(item_embeddings),
        )

        index.add(
            np.ascontiguousarray(
                item_embeddings[start:end],
                dtype=np.float32,
            )
        )

    D, I = index.search(
        np.ascontiguousarray(
            query_embeddings,
            dtype=np.float32,
        ),
        K,
    )

    predictions = {}
    scores = {}

    query_ids = queries.index.to_numpy()

    for pos, query_id in enumerate(query_ids):
        valid = I[pos] >= 0

        rows = I[pos][valid]
        row_scores = D[pos][valid]

        predictions[query_id] = item_ids[rows].tolist()

        scores[query_id] = row_scores.astype(float).tolist()

    del index

    return predictions, scores


def local_dense_search(
    queries,
    query_embeddings,
    items,
    item_embeddings,
    item_ids,
):
    location_to_rows = {
        location: group.index.to_numpy(dtype=np.int32)
        for location, group in items.groupby(
            "item_location_id",
            dropna=True,
        )
    }

    predictions = {}
    output_scores = {}

    for pos, (
        query_id,
        row,
    ) in enumerate(
        tqdm(
            queries.iterrows(),
            total=len(queries),
            desc="Local dense search",
        )
    ):
        location = row["search_location_id"]

        rows = location_to_rows.get(location)

        if rows is None or len(rows) == 0:
            predictions[query_id] = []
            output_scores[query_id] = []
            continue

        matrix = np.asarray(
            item_embeddings[rows],
            dtype=np.float32,
        )

        scores = matrix @ query_embeddings[pos]

        take_k = min(
            K,
            len(scores),
        )

        if len(scores) > take_k:
            top = np.argpartition(
                scores,
                -take_k,
            )[-take_k:]

            top = top[np.argsort(scores[top])[::-1]]
        else:
            top = np.argsort(scores)[::-1]

        selected_rows = rows[top]

        predictions[query_id] = item_ids[selected_rows].tolist()

        output_scores[query_id] = scores[top].astype(float).tolist()

    return predictions, output_scores


def dense_microcat_top1(
    queries,
    query_embeddings,
    items,
    item_embeddings,
    item_ids,
    priors,
):
    item_microcats = (
        pd.to_numeric(
            items["item_microcat_id"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )

    microcat_to_rows = defaultdict(list)

    for row_idx, microcat in enumerate(item_microcats):
        if microcat >= 0:
            microcat_to_rows[int(microcat)].append(row_idx)

    microcat_to_rows = {
        key: np.asarray(
            value,
            dtype=np.int32,
        )
        for key, value in microcat_to_rows.items()
    }

    predictions = {}
    output_scores = {}

    query_ids = queries.index.to_numpy()

    for pos, query_id in enumerate(
        tqdm(
            query_ids,
            desc="Dense microcat top1",
        )
    ):
        prior = priors.get(query_id, {})

        microcats = prior.get(
            "microcats",
            [],
        )

        if not microcats:
            predictions[query_id] = []
            output_scores[query_id] = []
            continue

        microcat = int(microcats[0])

        rows = microcat_to_rows.get(microcat)

        if rows is None or len(rows) == 0:
            predictions[query_id] = []
            output_scores[query_id] = []
            continue

        matrix = np.asarray(
            item_embeddings[rows],
            dtype=np.float32,
        )

        scores = matrix @ query_embeddings[pos]

        take_k = min(
            K,
            len(scores),
        )

        if len(scores) > take_k:
            top = np.argpartition(
                scores,
                -take_k,
            )[-take_k:]

            top = top[np.argsort(scores[top])[::-1]]
        else:
            top = np.argsort(scores)[::-1]

        selected_rows = rows[top]

        predictions[query_id] = item_ids[selected_rows].tolist()

        output_scores[query_id] = scores[top].astype(float).tolist()

    return predictions, output_scores


def main():
    train, benchmark_queries, benchmark_items = load_data(DATA_DIR)

    benchmark_items = benchmark_items.reset_index(drop=True)

    # ---------------------------------------------------------
    # Normalized queries
    # ---------------------------------------------------------

    if "search_query_norm" not in train.columns:
        train["search_query_norm"] = train["search_query"].map(normalize_query)

    if "search_query_norm" not in benchmark_queries.columns:
        benchmark_queries["search_query_norm"] = benchmark_queries["search_query"].map(
            normalize_query
        )

    queries = benchmark_queries.drop_duplicates("query_id").set_index("query_id")

    item_ids = benchmark_items["item_id"].to_numpy()

    print(
        "Benchmark queries:",
        len(queries),
    )

    print(
        "Benchmark items:",
        len(benchmark_items),
    )

    # =========================================================
    # 1. LOCAL WORD
    # =========================================================

    print()
    print("=" * 80)
    print("LOCAL WORD")
    print("=" * 80)

    word = LocalWordTfidfRetriever(
        ngram_range=(1, 2),
        min_df=2,
    )

    word.fit(benchmark_items)

    (
        word_pred,
        word_scores,
    ) = word.retrieve(
        queries,
        k=50,
        return_scores=True,
    )

    save_pickle(
        word_pred,
        PRED_DIR / "word_local_bench.pkl",
    )

    save_pickle(
        word_scores,
        PRED_DIR / "word_local_scores_bench.pkl",
    )

    del word
    gc.collect()

    # =========================================================
    # 2. LOCAL CHAR
    # =========================================================

    print()
    print("=" * 80)
    print("LOCAL CHAR")
    print("=" * 80)

    char = LocalCharTfidfRetriever(
        ngram_range=(3, 5),
        min_df=2,
    )

    char.fit(benchmark_items)

    (
        char_pred,
        char_scores,
    ) = char.retrieve(
        queries,
        k=50,
        return_scores=True,
    )

    save_pickle(
        char_pred,
        PRED_DIR / "char_local_bench.pkl",
    )

    save_pickle(
        char_scores,
        PRED_DIR / "char_local_scores_bench.pkl",
    )

    del char
    gc.collect()

    # =========================================================
    # 3. MICROCAT PRIOR FROM FULL TRAIN
    # =========================================================

    print()
    print("=" * 80)
    print("MICROCAT PRIOR")
    print("=" * 80)

    prior_model = MicrocatPrior(
        model_name=MODEL_NAME,
        device="cuda",
        encode_batch_size=128,
        max_seq_length=128,
        neighbor_k=30,
        top_microcats=20,
        similarity_power=4.0,
    )

    prior_model.fit(train)

    priors = prior_model.predict(queries)

    save_pickle(
        priors,
        FEATURE_DIR / "microcat_prior_bench.pkl",
    )

    del prior_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================
    # 4. BASE DENSE EMBEDDINGS
    # =========================================================

    print()
    print("=" * 80)
    print("BASE DENSE")
    print("=" * 80)

    dense = LocalDenseRetriever(
        model_name=MODEL_NAME,
        device="cuda",
        encode_batch_size=128,
        max_seq_length=128,
        fp16=True,
        query_batch_size=128,
    )

    dense.fit(
        benchmark_items,
        embeddings_path=(EMBED_DIR / "e5_base_benchmark_items.npy"),
        force_recompute=False,
    )

    # local dense
    (
        dense_local_pred,
        dense_local_scores,
    ) = dense.retrieve(
        queries,
        k=50,
    )

    save_pickle(
        dense_local_pred,
        PRED_DIR / "dense_local_bench.pkl",
    )

    save_pickle(
        dense_local_scores,
        PRED_DIR / "dense_local_scores_bench.pkl",
    )

    # query + params embeddings
    base_query_embeddings = dense._encode_queries(queries)

    # global dense
    (
        dense_global_pred,
        dense_global_scores,
    ) = global_dense_search(
        queries,
        base_query_embeddings,
        dense.item_embeddings,
        item_ids,
    )

    save_pickle(
        dense_global_pred,
        PRED_DIR / "dense_global_bench.pkl",
    )

    save_pickle(
        dense_global_scores,
        PRED_DIR / "dense_global_scores_bench.pkl",
    )

    # microcat-top1 dense
    (
        dense_microcat_pred,
        dense_microcat_scores,
    ) = dense_microcat_top1(
        queries,
        base_query_embeddings,
        benchmark_items,
        dense.item_embeddings,
        item_ids,
        priors,
    )

    save_pickle(
        dense_microcat_pred,
        PRED_DIR / "dense_microcat_bench.pkl",
    )

    save_pickle(
        dense_microcat_scores,
        PRED_DIR / "dense_microcat_scores_bench.pkl",
    )

    # =========================================================
    # 5. DESCRIPTION DENSE
    # =========================================================

    print()
    print("=" * 80)
    print("DESCRIPTION DENSE")
    print("=" * 80)

    dense.model.max_seq_length = 256

    desc_embeddings = build_desc_embeddings(
        dense.model,
        benchmark_items,
        EMBED_DIR / "e5_base_benchmark_items_desc.npy",
    )

    query_texts = build_dense_query_text(queries)

    desc_query_embeddings = dense.model.encode(
        query_texts.tolist(),
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(
        np.float32,
        copy=False,
    )

    # description local
    (
        desc_local_pred,
        desc_local_scores,
    ) = local_dense_search(
        queries,
        desc_query_embeddings,
        benchmark_items,
        desc_embeddings,
        item_ids,
    )

    save_pickle(
        desc_local_pred,
        PRED_DIR / "dense_desc_local_bench.pkl",
    )

    save_pickle(
        desc_local_scores,
        PRED_DIR / "dense_desc_local_scores_bench.pkl",
    )

    # description global
    (
        desc_global_pred,
        desc_global_scores,
    ) = global_dense_search(
        queries,
        desc_query_embeddings,
        desc_embeddings,
        item_ids,
    )

    save_pickle(
        desc_global_pred,
        PRED_DIR / "dense_desc_global_bench.pkl",
    )

    save_pickle(
        desc_global_scores,
        PRED_DIR / "dense_desc_global_scores_bench.pkl",
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print("All 7 benchmark sources saved.")


if __name__ == "__main__":
    main()

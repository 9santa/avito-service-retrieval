"""Compare BGE-M3 dense retrieval with the existing E5 validation artifacts.

Run from the repository root: python -m scripts.run_bge_m3_ablation
Each retrieval stage runs in a separate process to release RAM and GPU memory.
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.data import build_train_item_corpus
from src.fusion import rrf_fusion, union_predictions
from src.metrics import build_labels, recall_at_k


MODEL_NAME = "BAAI/bge-m3"
VAL_PATH = Path("data/interim/val_part.parquet")
BASELINE_DIR = Path("artifacts/predictions")
PREDICTION_FILES = {
    "char_local": "char_local_val.pkl",
    "word_local": "word_local_val.pkl",
    "dense_local": "dense_local_val.pkl",
    "dense_global": "dense_val.pkl",
    "dense_microcat": "dense_microcat_top1_val.pkl",
}
SCORE_FILES = {
    "dense_local": "dense_local_scores_val.pkl",
    "dense_global": "dense_scores_val.pkl",
    "dense_microcat": "dense_microcat_top1_scores_val.pkl",
}
STAGES = ("local", "global", "microcat", "evaluate")


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_retrieval_data():
    train = pd.read_parquet("data/raw/train.parquet")
    items = build_train_item_corpus(train)
    queries = (
        pd.read_parquet(VAL_PATH)[[
            "query_signature", "search_query_norm",
            "search_infm_params_text", "search_location_id",
        ]]
        .drop_duplicates("query_signature")
        .set_index("query_signature")
    )
    return items, queries


def run_retrieval(args):
    pred_dir = args.output_dir / "predictions"
    embeddings_path = args.output_dir / "embeddings" / "bge_m3_items.npy"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "microcat":
        from scripts.run_dense_microcat import main as run_microcat

        run_microcat(
            model_name=MODEL_NAME,
            encode_batch_size=args.batch_size,
            max_seq_length=128,
            query_prefix="",
            item_embeddings_path=embeddings_path,
            item_ids_cache_path=embeddings_path.with_name("bge_m3_items_item_ids.npy"),
            pred_dir=pred_dir,
            dense_local_path=pred_dir / PREDICTION_FILES["dense_local"],
            dense_global_path=pred_dir / PREDICTION_FILES["dense_global"],
            cutoffs=(1,),
        )
        return

    items, queries = load_retrieval_data()
    if args.stage == "local":
        from src.local_dense_retriever import LocalDenseRetriever

        retriever = LocalDenseRetriever(
            model_name=MODEL_NAME,
            encode_batch_size=args.batch_size,
            max_seq_length=128,
            fp16=True,
            item_prefix="",
            query_prefix="",
        )
        retriever.fit(items, embeddings_path=embeddings_path)
        predictions, scores = retriever.retrieve(queries, k=50)
        source = "dense_local"
    else:
        from src.dense_retriever import GlobalDenseRetriever

        retriever = GlobalDenseRetriever(
            model_name=MODEL_NAME,
            batch_size=args.batch_size,
            max_seq_len=128,
            fp16=True,
            item_prefix="",
            query_prefix="",
        )
        retriever.fit(items, embeddings_path=embeddings_path)
        predictions, scores = retriever.retrieve(queries, k=50, return_scores=True)
        source = "dense_global"

    save_pickle(pred_dir / PREDICTION_FILES[source], predictions)
    save_pickle(pred_dir / SCORE_FILES[source], scores)


def compare_predictions(labels, baseline, experiment):
    """Compare matched sources and fusions on exactly the same query IDs."""
    rows = []

    def compare(name, before, after, cutoffs=(10, 20, 50)):
        for k in cutoffs:
            old = recall_at_k(labels, before, k=k)
            new = recall_at_k(labels, after, k=k)
            rows.append({
                "retrieval": name,
                "metric": "pool_recall" if name == "full_pool" else f"recall@{k}",
                "e5": old,
                "bge_m3": new,
                "delta_pp": 100 * (new - old),
                "relative_gain_pct": 100 * (new / old - 1) if old else None,
            })

    for source in SCORE_FILES:
        compare(source, baseline[source], experiment[source])
    for name, sources in [
        ("rrf_local", ["dense_local", "char_local", "word_local"]),
        ("rrf_all", ["dense_local", "char_local", "word_local", "dense_global", "dense_microcat"]),
    ]:
        before = rrf_fusion([baseline[source] for source in sources], k=50, rrf_k=60)
        after = rrf_fusion([experiment[source] for source in sources], k=50, rrf_k=60)
        compare(name, before, after)

    before_pool = union_predictions(list(baseline.values()))
    after_pool = union_predictions(list(experiment.values()))
    compare("full_pool", before_pool, after_pool, cutoffs=(1000,))
    return pd.DataFrame(rows)


def evaluate(output_dir):
    labels = build_labels(pd.read_parquet(VAL_PATH), query_col="query_signature")
    baseline = {}
    experiment = {}
    for source, filename in PREDICTION_FILES.items():
        baseline[source] = load_pickle(BASELINE_DIR / filename)
        experiment[source] = (
            baseline[source] if source in ("char_local", "word_local")
            else load_pickle(output_dir / "predictions" / filename)
        )
        for name, predictions in [("E5", baseline[source]), ("BGE-M3", experiment[source])]:
            missing = set(labels) - predictions.keys()
            if missing:
                raise ValueError(f"{name} {source}: missing {len(missing)} validation queries")
        # Match query coverage and ordering, including for fusion's first source.
        baseline[source] = {query: baseline[source][query] for query in labels}
        experiment[source] = {query: experiment[source][query] for query in labels}

    table = compare_predictions(labels, baseline, experiment)
    table.to_csv(output_dir / "comparison.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved: {output_dir / 'comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablations/bge_m3"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == Path("artifacts").resolve():
        parser.error("Use a separate experiment directory to preserve E5 artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": MODEL_NAME,
        "max_seq_length": 128,
        "item_prefix": "",
        "query_prefix": "",
        "fp16": True,
        "k": 50,
        "rrf_k": 60,
        "microcat_cutoff": 1,
        "microcat_prior": "artifacts/features/microcat_prior_val.pkl",
        "validation": str(VAL_PATH),
        "baseline_dir": str(BASELINE_DIR),
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Experiment configuration changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    if args.stage == "all":
        for stage in STAGES:
            print(f"Running {stage}; log: {args.output_dir / (stage + '.log')}", flush=True)
            with (args.output_dir / f"{stage}.log").open("w") as log:
                subprocess.run([
                    sys.executable, "-u", "-m", "scripts.run_bge_m3_ablation",
                    "--stage", stage, "--output-dir", str(args.output_dir),
                    "--batch-size", str(args.batch_size),
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
        print((args.output_dir / "comparison.csv").read_text())
        return

    start = time.perf_counter()
    if args.stage == "evaluate":
        evaluate(args.output_dir)
    else:
        run_retrieval(args)
    elapsed = time.perf_counter() - start
    (args.output_dir / f"{args.stage}_timing.json").write_text(
        json.dumps({"seconds": elapsed, "batch_size": args.batch_size}, indent=2) + "\n"
    )
    print(f"{args.stage}: {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()

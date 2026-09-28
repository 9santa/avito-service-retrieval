"""Загрузка исходных parquet-файлов и построение корпуса объявлений для локальной валидации."""

from pathlib import Path

import pandas as pd


SEARCH_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]


def load_data(data_dir: str | Path = "data/raw"):
    """
    Загружает train, benchmark queries и benchmark items.

    Parameters
    ----------
    data_dir: str | Path
        Папка с сырыми parquet файлами.

    Returns
    -------
    train: pd.DataFrame
    queries: pd.DataFrame
    items: pd.DataFrame
    """
    data_dir = Path(data_dir)

    train = pd.read_parquet(data_dir / "train.parquet")
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")

    return train, queries, items


def build_train_item_corpus(train: pd.DataFrame):
    """
    Строит локальный корпус объявленний для validation.

    Один item_id оставляем один раз.
    """
    return train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)

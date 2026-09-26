import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SEARCH_SIGNATURE_COLS = [
    "search_query_norm",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def normalize_query(text):
    """
    Нормализация поискового запроса.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_query_signature(df: pd.DataFrame):
    """
    Создает идентификатор (сигнатуру) уникальной комбинаци признаков запроса.
    """
    df = df.copy()

    if "search_query_norm" not in df.columns:
        df["search_query_norm"] = df["search_query"].map(normalize_query)

    signature_df = df[SEARCH_SIGNATURE_COLS].astype("string").fillna("__NA__")

    df["query_signature"] = pd.util.hash_pandas_object(
        signature_df,
        index=False,
    ).astype("uint64")

    return df


def make_validation_split(
    train: pd.DataFrame,
    val_size=0.2,
    random_state=42,
):
    """
    Разделяет train по нормализованному тексту запроса.

    Один search_query_norm целиком попадает либо в train, либо в validation.
    """
    df = add_query_signature(train)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["search_query_norm"],
        )
    )

    train_part = df.iloc[train_idx].copy()
    val_part = df.iloc[val_idx].copy()

    return train_part, val_part

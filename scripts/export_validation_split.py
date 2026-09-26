from pathlib import Path

from src.data import load_data
from src.validation import make_validation_split


DATA_DIR = "data/raw"
OUTPUT_DIR = Path("data/interim")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, _, _ = load_data(DATA_DIR)

    train_part, val_part = make_validation_split(
        train,
        val_size=0.2,
        random_state=42,
    )

    train_part.to_parquet(
        OUTPUT_DIR / "train_part.parquet",
        index=False,
    )

    val_part.to_parquet(
        OUTPUT_DIR / "val_part.parquet",
        index=False,
    )

    print("Train rows:", len(train_part))
    print("Validation rows:", len(val_part))

    print(
        "Train queries:",
        train_part["search_query_norm"].nunique(),
    )

    print(
        "Validation queries:",
        val_part["search_query_norm"].nunique(),
    )

    assert set(train_part["search_query_norm"]).isdisjoint(
        set(val_part["search_query_norm"])
    )


if __name__ == "__main__":
    main()

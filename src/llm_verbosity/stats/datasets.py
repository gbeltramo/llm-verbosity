import random
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")

DATASETS = {
    "math500": "math500.parquet",
    "gpqa": "gpqa_diamond.parquet",
    "hle": "hle_diamond.parquet",
}


def load_datasets(
    datasets: list[str],
    data_dir: Path = DATA_DIR,
) -> dict[str, pl.DataFrame]:
    """Load selected datasets from Parquet files."""
    loaded = {}

    for tag in datasets:
        if tag not in DATASETS:
            raise ValueError(f"Unknown dataset: {tag}")

        path = data_dir / DATASETS[tag]

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        loaded[tag] = pl.read_parquet(path)

    return loaded


def select_indices(
    len_subset: int,
    len_dataset: int,
    offset: int = 0,
    random_seed: int = 1,
) -> list[int]:
    """Select indices without replacement."""
    if len_subset < 0 or offset < 0:
        raise ValueError("size_subset and offset must be non-negative")

    available = list(range(offset, len_dataset))

    if len_subset > len(available):
        raise ValueError(f"{len_subset=} invalid. May need to decrease `offset`")

    if not available or len_subset == 0:
        return []

    return random.Random(random_seed).sample(available, len_subset)


def build_prompts(
    datasets: list[str],
    len_subset: int,
    offset: int = 0,
    data_dir: Path = DATA_DIR,
    random_seed: int = 1,
) -> dict[str, list[str]]:
    """Return up to `len_subset` prompts for each dataset."""
    loaded = load_datasets(datasets, data_dir)

    prompts: dict[str, list[str]] = dict()

    for tag, df in loaded.items():
        indices = select_indices(
            len_subset=len_subset,
            len_dataset=len(df),
            offset=offset,
            random_seed=random_seed,
        )

        prompts[tag] = df["question"][indices].to_list()

    return prompts

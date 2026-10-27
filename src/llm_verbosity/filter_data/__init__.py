"""Dataset preparation helpers (kept import-light: no heavy deps here)."""

# tag -> (parquet file name, difficulty)
DATASETS: dict[str, tuple[str, str]] = {
    "MATH-500": ("math500.parquet", "easy"),
    "GPQA-Diamond": ("gpqa_diamond.parquet", "mid"),
    "HLE-Diamond": ("hle_diamond.parquet", "hard"),
}

COLUMNS = ["id", "question", "image", "answer", "category"]

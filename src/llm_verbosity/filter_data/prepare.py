"""Download MATH-500 / GPQA-Diamond / HLE and write 100-question subsets as zstd parquet.

By default (--text-only) any question that carries non-text media (image, audio, video) is
dropped BEFORE sampling, so you still get n text-only questions per dataset.

GPQA and HLE are gated on the Hugging Face Hub: accept their terms on the dataset pages
and export HF_TOKEN (or run `huggingface-cli login`) before running this script.
"""

import argparse
import random
import sys
from pathlib import Path

import polars as pl
from datasets import load_dataset

from llm_verbosity.filter_data import COLUMNS, DATASETS

# A column is considered "question-side media" if its name contains one of these hints...
MEDIA_HINTS = ("image", "audio", "video")
# ...unless it also contains one of these (media attached to the rationale, not the question).
MEDIA_IGNORE = ("rationale",)


def _non_empty(name: str, dtype: pl.DataType) -> pl.Expr:
    """Expression that is True when the cell holds actual content."""
    col = pl.col(name)
    if dtype == pl.String:
        return col.is_not_null() & (col.str.strip_chars() != "")
    if isinstance(dtype, pl.List):
        return col.is_not_null() & (col.list.len() > 0)
    if isinstance(dtype, pl.Struct):  # e.g. datasets' Image/Audio features: {bytes, path}
        fields = [f.name for f in dtype.fields]
        return pl.any_horizontal(*[col.struct.field(f).is_not_null() for f in fields])
    return col.is_not_null()


def media_columns(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if any(h in c.lower() for h in MEDIA_HINTS) and not any(i in c.lower() for i in MEDIA_IGNORE)
    ]


def drop_non_text(df: pl.DataFrame, name: str, text_only: bool) -> pl.DataFrame:
    if not text_only:
        return df
    cols = media_columns(df)
    if not cols:
        print(f"[{name}] no media columns found -> all {df.height} questions are text-only")
        return df
    has_media = pl.any_horizontal(*[_non_empty(c, df.schema[c]) for c in cols])
    kept = df.filter(~has_media)
    print(f"[{name}] text-only filter on columns {cols}: dropped {df.height - kept.height}, kept {kept.height}")
    return kept


def _sample(df: pl.DataFrame, n: int, seed: int, name: str) -> pl.DataFrame:
    if df.height < n:
        print(f"[WARN] {name}: only {df.height} questions available (< {n}); using all of them")
        return df
    return df.sample(n=n, seed=seed)


# --------------------------------------------------------------------------- datasets


def prepare_math500(n: int, seed: int, text_only: bool) -> pl.DataFrame:
    df = load_dataset("HuggingFaceH4/MATH-500", split="test").to_polars()
    df = _sample(drop_non_text(df, "MATH-500", text_only), n, seed, "MATH-500")
    return df.select(
        pl.col("unique_id").cast(pl.String).alias("id"),
        pl.col("problem").alias("question"),
        pl.lit("").alias("image"),
        pl.col("answer").cast(pl.String),
        pl.col("subject").alias("category"),
    )


def prepare_gpqa_diamond(n: int, seed: int, text_only: bool) -> pl.DataFrame:
    df = load_dataset("Idavidrein/GPQA", "gpqa_diamond", split="train").to_polars()
    df = _sample(drop_non_text(df, "GPQA-Diamond", text_only), n, seed, "GPQA-Diamond")

    letters = "ABCD"
    rows = []
    for i, r in enumerate(df.iter_rows(named=True)):
        record_id = str(r.get("Record ID") or f"row{i}")
        options = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        options = [str(o).strip() for o in options]
        # Deterministic per-question shuffle of the 4 options.
        order = list(range(4))
        random.Random(f"{seed}-{record_id}").shuffle(order)
        body = "\n".join(f"{letters[pos]}) {options[j]}" for pos, j in enumerate(order))
        question = f"{str(r['Question']).strip()}\n\n{body}\n\nAnswer with the letter of the correct option."
        rows.append(
            {
                "id": record_id,
                "question": question,
                "image": "",
                "answer": letters[order.index(0)],  # position of the correct option
                "category": str(r.get("High-level domain") or ""),
            }
        )
    return pl.DataFrame(rows, schema={c: pl.String for c in COLUMNS})


def prepare_hle_diamond(n: int, seed: int, repo: str, split: str, text_only: bool) -> pl.DataFrame:
    df = load_dataset(repo, split=split).to_polars()
    df = _sample(drop_non_text(df, "HLE-Diamond", text_only), n, seed, "HLE-Diamond")

    if "image" in df.columns and df.schema["image"] == pl.String:
        image = pl.col("image").fill_null("")
    elif text_only or "image" not in df.columns:
        image = pl.lit("")
    else:
        sys.exit(
            f"HLE column 'image' has dtype {df.schema['image']}, expected a string "
            "(data URL / http URL). Re-run with --text-only."
        )
    if not text_only:
        print("[WARN] HLE: multi-modal questions kept; the target model must accept image inputs")

    return df.select(
        pl.col("id").cast(pl.String),
        pl.col("question"),
        image.alias("image"),
        pl.col("answer").cast(pl.String),
        pl.col("category").cast(pl.String),
    )


def _write(df: pl.DataFrame, path: Path) -> None:
    df = df.select(COLUMNS)
    df.write_parquet(path, compression="zstd", compression_level=9)
    print(f"wrote {df.height:>4} rows -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=100, help="questions per dataset (default: 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop questions with image/audio/video content (default: on; --no-text-only to keep)",
    )
    p.add_argument("--hle-repo", default="cais/hle", help="HF dataset repo used for HLE")
    p.add_argument("--hle-split", default="test")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def out(tag: str) -> Path:
        return args.output_dir / DATASETS[tag][0]

    _write(prepare_math500(args.n, args.seed, args.text_only), out("MATH-500"))
    _write(prepare_gpqa_diamond(args.n, args.seed, args.text_only), out("GPQA-Diamond"))
    _write(
        prepare_hle_diamond(args.n, args.seed, args.hle_repo, args.hle_split, args.text_only),
        out("HLE-Diamond"),
    )


if __name__ == "__main__":
    main()

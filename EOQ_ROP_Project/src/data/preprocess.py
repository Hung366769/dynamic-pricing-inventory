"""Load, validate, and preprocess the Retail Price Optimization dataset."""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "retail_price.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "retail_price_clean.csv"

NUMERIC_COLUMNS = [
    "qty", "total_price", "freight_price", "unit_price", "product_name_lenght",
    "product_description_lenght", "product_photos_qty", "product_weight_g",
    "product_score", "customers", "weekday", "weekend", "holiday", "month",
    "year", "s", "volume", "comp_1", "ps1", "fp1", "comp_2", "ps2",
    "fp2", "comp_3", "ps3", "fp3", "lag_price",
]
KEY_COLUMNS = ["product_id", "month_year", "unit_price", "qty"]


def load_and_clean(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Return a validated dataframe while preserving source rows and columns.

    Date parsing follows the source format DD-MM-YYYY. Duplicate product-month
    keys are reported but not silently aggregated, because aggregation could
    change the economic meaning of a row.
    """
    input_path = Path(input_path)
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()

    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

    df["date"] = pd.to_datetime(df["month_year"], format="%d-%m-%Y", errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove only rows unusable for the price-demand analysis.
    before = len(df)
    df = df.dropna(subset=["product_id", "date", "unit_price", "qty"]).copy()
    df = df[(df["unit_price"] > 0) & (df["qty"] > 0)].copy()
    dropped = before - len(df)

    duplicated = int(df.duplicated(["product_id", "date"]).sum())
    if duplicated:
        print(
            f"WARNING: found {duplicated} duplicate product-date keys. "
            "They were retained for inspection; review before model fitting."
        )

    competitor_cols = [c for c in ["comp_1", "comp_2", "comp_3"] if c in df.columns]
    if competitor_cols:
        df["competitor_price"] = df[competitor_cols].replace(0, pd.NA).mean(axis=1)
    else:
        df["competitor_price"] = pd.NA

    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["month_num"] = df["date"].dt.month
    df["year_num"] = df["date"].dt.year
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    print(f"Input rows: {before}; retained rows: {len(df)}; removed invalid rows: {dropped}")
    print(f"Unique products: {df['product_id'].nunique()}; unique months: {df['year_month'].nunique()}")
    print(f"Duplicate product-date keys: {duplicated}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to source CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path for cleaned CSV")
    args = parser.parse_args()

    df = load_and_clean(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"Saved clean dataset to: {out}")


if __name__ == "__main__":
    main()

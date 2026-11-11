"""Generate a compact EDA report and figures for the supplied dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/retail_price_clean.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = ["qty", "unit_price", "total_price", "comp_1", "comp_2", "comp_3", "lag_price"]
    for col in numeric:
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    comp_cols = [c for c in ["comp_1", "comp_2", "comp_3"] if c in df]
    df["competitor_price"] = df[comp_cols].replace(0, np.nan).mean(axis=1)
    by_product = df.groupby("product_id")
    price_var = by_product["unit_price"].nunique()
    report = {
        "rows": int(len(df)),
        "columns_in_clean_file": int(len(df.columns)),
        "raw_source_columns": 30,
        "products": int(df["product_id"].nunique()),
        "categories": int(df["product_category_name"].nunique()),
        "months": int(df["date"].dt.to_period("M").nunique()),
        "date_min": df["date"].min().strftime("%Y-%m-%d"),
        "date_max": df["date"].max().strftime("%Y-%m-%d"),
        "missing_cells_source_columns": int(df[[c for c in df.columns if c not in ["date", "year_month", "month_num", "year_num", "competitor_price"]]].isna().sum().sum()),
        "duplicate_product_month_keys": int(df.duplicated(["product_id", "date"]).sum()),
        "products_with_varying_price_full_period": int((price_var > 1).sum()),
        "products_total": int(len(price_var)),
        "qty_min": float(df["qty"].min()),
        "qty_median": float(df["qty"].median()),
        "qty_max": float(df["qty"].max()),
        "unit_price_min": float(df["unit_price"].min()),
        "unit_price_median": float(df["unit_price"].median()),
        "unit_price_max": float(df["unit_price"].max()),
        "total_price_qty_unit_price_mismatch_rows_over_0_01": int((np.abs(df["total_price"] - df["qty"]*df["unit_price"]) > 0.01).sum()),
    }
    cutoff = pd.Timestamp("2018-04-01")
    train = df[df["date"] <= cutoff]
    train_price_nunique = train.groupby("product_id")["unit_price"].nunique()
    report["default_train_rows_through_2018_04"] = int(len(train))
    report["default_test_rows_2018_05_to_2018_08"] = int(len(df)-len(train))
    report["products_with_varying_price_in_train"] = int((train_price_nunique > 1).sum())
    report["products_with_single_price_in_train"] = int((train_price_nunique == 1).sum())

    # Check the lag_price feature against the previous observed month for the same product.
    data = df.sort_values(["product_id", "date"])
    eligible = matches = 0
    for _, group in data.groupby("product_id"):
        group = group.sort_values("date")
        prevs = list(group.itertuples(index=False))
        for a, b in zip(prevs, prevs[1:]):
            months = (b.date.year-a.date.year)*12 + b.date.month-a.date.month
            if months == 1:
                eligible += 1
                if abs(float(b.lag_price)-float(a.unit_price)) <= 0.03:
                    matches += 1
    report["consecutive_month_pairs_for_lag_check"] = eligible
    report["lag_price_matches_previous_month_unit_price"] = matches

    with open(OUT / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    monthly = df.groupby(df["date"].dt.to_period("M")).agg(rows=("product_id", "size"), products=("product_id", "nunique"), mean_price=("unit_price", "mean"), median_qty=("qty", "median")).reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly.to_csv(OUT / "monthly_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["date"], monthly["products"], marker="o")
    ax.set_title("Số sản phẩm xuất hiện theo tháng")
    ax.set_xlabel("Tháng"); ax.set_ylabel("Số sản phẩm")
    fig.tight_layout(); fig.savefig(FIG / "products_by_month.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["unit_price"], df["qty"], alpha=0.55)
    ax.set_title("Giá bán và lượng bán quan sát")
    ax.set_xlabel("unit_price"); ax.set_ylabel("qty")
    fig.tight_layout(); fig.savefig(FIG / "price_vs_qty.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["unit_price"].dropna(), bins=25)
    ax.set_title("Phân phối giá bán")
    ax.set_xlabel("unit_price"); ax.set_ylabel("Số quan sát")
    fig.tight_layout(); fig.savefig(FIG / "unit_price_distribution.png", dpi=140); plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

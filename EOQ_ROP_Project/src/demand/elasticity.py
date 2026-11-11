"""Estimate observational price elasticity with a log-log panel regression.

The price coefficient is an association conditional on the included controls,
not automatically a causal elasticity: prices may be endogenous to expected
sales, promotions, or unobserved demand shocks.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "retail_price_clean.csv"
DEFAULT_OUT = ROOT / "outputs"


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    elif "month_year" in data.columns:
        data["date"] = pd.to_datetime(data["month_year"], format="%d-%m-%Y", errors="coerce")
    else:
        raise ValueError("Need a parsed 'date' column or original 'month_year'.")
    for c in ["qty", "unit_price"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    comp_cols = [c for c in ["comp_1", "comp_2", "comp_3"] if c in data.columns]
    if "competitor_price" not in data.columns:
        if not comp_cols:
            raise ValueError("No competitor-price columns found.")
        data["competitor_price"] = data[comp_cols].replace(0, np.nan).mean(axis=1)
    data["competitor_price"] = pd.to_numeric(data["competitor_price"], errors="coerce")
    data = data.dropna(subset=["product_id", "date", "qty", "unit_price", "competitor_price"])
    data = data[(data["qty"] > 0) & (data["unit_price"] > 0) & (data["competitor_price"] > 0)].copy()
    data["log_qty"] = np.log(data["qty"])
    data["log_price"] = np.log(data["unit_price"])
    data["log_competitor_price"] = np.log(data["competitor_price"])
    data["month_num"] = data["date"].dt.month
    data["year_num"] = data["date"].dt.year
    return data.sort_values(["date", "product_id"]).reset_index(drop=True)


def fit_elasticity(df: pd.DataFrame, train_end: str = "2018-04-01") -> tuple[object, dict, pd.DataFrame]:
    """Fit a train-period fixed-effects regression and evaluate on later months."""
    data = prepare_model_frame(df)
    cutoff = pd.Timestamp(train_end)
    train = data[data["date"] <= cutoff].copy()
    test = data[data["date"] > cutoff].copy()
    if len(train) < 50:
        raise ValueError(f"Too few training rows ({len(train)}); check train_end.")

    formula = "log_qty ~ log_price + log_competitor_price + C(product_id) + C(month_num) + C(year_num)"
    model = smf.ols(formula=formula, data=train).fit(
        cov_type="cluster", cov_kwds={"groups": train["product_id"]}
    )
    coefficient = float(model.params["log_price"])
    ci_low, ci_high = map(float, model.conf_int().loc["log_price"].tolist())

    # Out-of-time prediction is evaluated only for products seen in training.
    # The training categories for month/year must be represented in the train period.
    test_known = test[test["product_id"].isin(train["product_id"].unique())].copy()
    metrics = {"test_rows": int(len(test_known)), "test_rows_excluded_unseen_products": int(len(test) - len(test_known))}
    if len(test_known):
        pred_log = np.asarray(model.predict(test_known), dtype=float)
        # Duan smearing corrects the systematic bias from exponentiating log predictions.
        smear = float(np.mean(np.exp(np.asarray(model.resid, dtype=float))))
        pred_qty = np.maximum(0.0, np.exp(pred_log) * smear)
        actual = test_known["qty"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - pred_qty)))
        rmse = float(np.sqrt(np.mean((actual - pred_qty) ** 2)))
        wape = float(np.sum(np.abs(actual - pred_qty)) / max(np.sum(actual), 1e-12) * 100)
        smape = float(np.mean(2 * np.abs(actual - pred_qty) / np.maximum(np.abs(actual) + np.abs(pred_qty), 1e-12)) * 100)
        # A simple, hard-to-beat benchmark: each known product's median qty in training.
        product_median = train.groupby("product_id")["qty"].median()
        naive_pred = test_known["product_id"].map(product_median).to_numpy(dtype=float)
        naive_mae = float(np.mean(np.abs(actual - naive_pred)))
        naive_rmse = float(np.sqrt(np.mean((actual - naive_pred) ** 2)))
        naive_wape = float(np.sum(np.abs(actual - naive_pred)) / max(np.sum(actual), 1e-12) * 100)
        naive_smape = float(np.mean(2 * np.abs(actual - naive_pred) / np.maximum(np.abs(actual) + np.abs(naive_pred), 1e-12)) * 100)
        metrics.update({
            "mae_qty": mae, "rmse_qty": rmse, "wape_percent": wape, "smape_percent": smape,
            "naive_product_median_mae_qty": naive_mae,
            "naive_product_median_rmse_qty": naive_rmse,
            "naive_product_median_wape_percent": naive_wape,
            "naive_product_median_smape_percent": naive_smape,
        })
        predictions = test_known[["product_id", "date", "qty", "unit_price", "competitor_price"]].copy()
        predictions["predicted_qty"] = pred_qty
        predictions["absolute_error"] = np.abs(actual - pred_qty)
    else:
        metrics.update({"mae_qty": None, "rmse_qty": None, "wape_percent": None, "smape_percent": None})
        predictions = pd.DataFrame()

    summary = {
        "train_end": cutoff.strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "test_rows_total": int(len(test)),
        "train_products": int(train["product_id"].nunique()),
        "elasticity_estimate": coefficient,
        "elasticity_ci95_low": ci_low,
        "elasticity_ci95_high": ci_high,
        "elasticity_p_value_clustered_by_product": float(model.pvalues["log_price"]),
        "r_squared_train": float(model.rsquared),
        "adjusted_r_squared_train": float(model.rsquared_adj),
        "formula": formula,
        "metrics": metrics,
        "interpretation_warning": (
            "Observational log-log price coefficient with product, month and year controls. "
            "Do not interpret as causal without a credible identification strategy."
        ),
    }
    return model, summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--train-end", default="2018-04-01", help="Inclusive end date for training")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    _, summary, predictions = fit_elasticity(df, train_end=args.train_end)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "elasticity_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    predictions.to_csv(out / "elasticity_holdout_predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

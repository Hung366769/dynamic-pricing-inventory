"""Chronological holdout evaluation against non-RL pricing baselines."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from src.demand.elasticity import fit_elasticity
from src.evaluation.baselines import evaluate_policies

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/retail_price_clean.csv"))
    parser.add_argument("--train-end", default="2018-04-01")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--q-table", default=str(ROOT / "artifacts/q_table.npy"))
    parser.add_argument("--ordering-cost", type=float, default=25.0)
    parser.add_argument("--annual-holding-cost-rate", type=float, default=0.20)
    parser.add_argument("--lead-time-periods", type=int, default=1)
    parser.add_argument("--safety-stock-units", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train = df[df["date"] <= pd.Timestamp(args.train_end)].copy()
    test = df[df["date"] > pd.Timestamp(args.train_end)].copy()
    _, elasticity_result, _ = fit_elasticity(df, train_end=args.train_end)
    q_path = Path(args.q_table)
    q_table = np.load(q_path) if q_path.exists() else None
    detail, summary = evaluate_policies(train, test, elasticity_result["elasticity_estimate"], q_table=q_table)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "policy_evaluation_detail.csv", index=False)
    summary.to_csv(out / "policy_evaluation_summary.csv", index=False)
    with open(out / "evaluation_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_end_inclusive": args.train_end,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "elasticity_estimate_used": elasticity_result["elasticity_estimate"],
            "evaluation_type": "historical replay counterfactual simulation",
            "caveat": "Observed held-out qty anchors demand at observed price, then demand is scaled by estimated elasticity. Inventory and unit-cost assumptions are simulated; results are not causal or real-world performance estimates.",
        }, f, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    main()

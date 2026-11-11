"""Run a small EOQ/ROP parameter sensitivity analysis on the holdout period."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.demand.elasticity import fit_elasticity
from src.evaluation.baselines import evaluate_policies

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT/"data/processed/retail_price_clean.csv"))
    ap.add_argument("--train-end", default="2018-04-01")
    ap.add_argument("--out", default=str(ROOT/"outputs/eoq_sensitivity.csv"))
    ap.add_argument("--q-table", default=str(ROOT/"artifacts/q_table.npy"))
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train = df[df.date <= pd.Timestamp(args.train_end)].copy()
    test = df[df.date > pd.Timestamp(args.train_end)].copy()
    _, result, _ = fit_elasticity(df, train_end=args.train_end)
    q_path = Path(args.q_table)
    q = np.load(q_path) if q_path.exists() else None
    rows = []
    for s in (10.0, 25.0, 50.0):
        for h_rate in (0.10, 0.20, 0.30):
            for lead in (1, 2):
                detail, summary = evaluate_policies(
                    train, test, float(result["elasticity_estimate"]), q_table=q,
                    ordering_cost=s, annual_holding_cost_rate=h_rate,
                    lead_time_periods=lead, safety_stock_units=0.0,
                )
                summary = summary[summary.policy.isin(["fixed_qty_rop", "eoq_rop"])]
                for _, row in summary.iterrows():
                    rows.append({
                        "ordering_cost_assumption": s,
                        "annual_holding_cost_rate_assumption": h_rate,
                        "lead_time_months_assumption": lead,
                        **row.to_dict()
                    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} sensitivity rows to {out}")

if __name__ == "__main__":
    main()

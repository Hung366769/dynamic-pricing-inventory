"""Repeat Q-learning across random seeds and aggregate policy results."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.demand.elasticity import fit_elasticity
from src.environment.inventory_env import InventoryPricingEnv
from src.rl.q_learning import train_q_learning
from src.evaluation.baselines import evaluate_policies

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/retail_price_clean.csv"))
    parser.add_argument("--train-end", default="2018-04-01")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per seed")
    parser.add_argument("--seeds", default="42,7,21,2025,2026")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train = df[df["date"] <= pd.Timestamp(args.train_end)].copy()
    test = df[df["date"] > pd.Timestamp(args.train_end)].copy()
    _, elasticity_result, _ = fit_elasticity(df, train_end=args.train_end)
    elasticity = float(elasticity_result["elasticity_estimate"])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    all_summary, convergence = [], []
    for seed in seeds:
        env = InventoryPricingEnv(train, elasticity=elasticity, seed=seed)
        q, rewards, info = train_q_learning(env, episodes=args.episodes, seed=seed)
        _, summary = evaluate_policies(train, test, elasticity, q_table=q)
        summary["seed"] = seed
        all_summary.append(summary)
        convergence.append({
            "seed": seed, "episodes": args.episodes,
            "mean_reward_first_100": float(np.mean(rewards[:min(100, len(rewards))])),
            "mean_reward_last_100": float(np.mean(rewards[-min(100, len(rewards)):])) ,
            "max_reward": float(np.max(rewards)),
            "epsilon_final": info["epsilon_final"],
        })
        print(f"Completed seed={seed}; last-100 train reward={convergence[-1]['mean_reward_last_100']:.2f}")

    details = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    details.to_csv(out / "multi_seed_policy_results.csv", index=False)
    pd.DataFrame(convergence).to_csv(out / "multi_seed_convergence.csv", index=False)
    metrics = [c for c in ["simulated_revenue", "simulated_gross_profit", "simulated_total_reward",
                           "simulated_units_sold", "simulated_ending_inventory", "simulated_unmet_demand",
                           "simulated_stockout_rate_pct", "simulated_price_change_abs_mean"] if c in details.columns]
    aggregate = details.groupby("policy")[metrics].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(str(x) for x in tup if x != "") if isinstance(tup, tuple) else str(tup) for tup in aggregate.columns]
    aggregate.to_csv(out / "multi_seed_policy_summary.csv", index=False)
    metadata = {
        "train_end": args.train_end, "train_rows": int(len(train)), "test_rows": int(len(test)),
        "elasticity_used": elasticity, "seeds": seeds, "episodes_per_seed": args.episodes,
        "note": "All policy metrics are from the same historical-replay counterfactual simulator and remain conditional on the documented assumptions."
    }
    with open(out / "multi_seed_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("\nMean results by policy:")
    if not aggregate.empty:
        print(aggregate[[c for c in aggregate.columns if "simulated_total_reward_mean" in c or c == "policy"]].to_string(index=False))


if __name__ == "__main__":
    main()

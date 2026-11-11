"""Tabular Q-learning for the discrete pricing/inventory state space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.preprocess import load_and_clean
from src.demand.elasticity import fit_elasticity
from src.environment.inventory_env import InventoryPricingEnv, ACTIONS, STATE_DIMS

ROOT = Path(__file__).resolve().parents[2]


def train_q_learning(
    env: InventoryPricingEnv,
    episodes: int = 3000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.02,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> tuple[np.ndarray, list[float], dict]:
    rng = np.random.default_rng(seed)
    q = np.zeros((*STATE_DIMS, len(ACTIONS)), dtype=float)
    rewards: list[float] = []
    eps = float(epsilon)
    for episode in range(int(episodes)):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            if rng.random() < eps:
                action = int(rng.integers(len(ACTIONS)))
            else:
                action = int(np.argmax(q[state]))
            next_state, reward, done, _ = env.step(action)
            current = q[state + (action,)]
            target = reward if done or next_state is None else reward + gamma * float(np.max(q[next_state]))
            q[state + (action,)] = current + alpha * (target - current)
            state = next_state if next_state is not None else state
            total += reward
        rewards.append(float(total))
        eps = max(epsilon_min, eps * epsilon_decay)
    info = {
        "episodes": int(episodes), "alpha": alpha, "gamma": gamma,
        "epsilon_final": eps, "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay, "state_dims": STATE_DIMS,
        "actions": ACTIONS.tolist(), "seed": seed,
    }
    return q, rewards, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/retail_price_clean.csv"))
    parser.add_argument("--train-end", default="2018-04-01")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train_df = df[df["date"] <= pd.Timestamp(args.train_end)].copy()
    _, elasticity_result, _ = fit_elasticity(df, train_end=args.train_end)
    elasticity = elasticity_result["elasticity_estimate"]
    env = InventoryPricingEnv(train_df, elasticity=elasticity, seed=args.seed)
    q, rewards, training_info = train_q_learning(env, episodes=args.episodes, seed=args.seed)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_outputs = Path(args.outputs_dir); out_outputs.mkdir(parents=True, exist_ok=True)
    np.save(out / "q_table.npy", q)
    training_info.update({
        "elasticity_used": elasticity,
        "train_end": args.train_end,
        "simulator_assumptions": {
            "cost_ratio": env.cost_ratio,
            "holding_cost_rate": env.holding_cost_rate,
            "stockout_penalty_rate": env.stockout_penalty_rate,
            "price_change_penalty_rate": env.price_change_penalty_rate,
            "replenishment_rate": env.replenishment_rate,
            "max_price_deviation": env.max_price_deviation,
            "noise_sigma": env.noise_sigma,
        },
    })
    with open(out / "q_learning_config.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"episode": np.arange(1, len(rewards)+1), "total_reward": rewards}).to_csv(out_outputs / "training_rewards.csv", index=False)
    print(f"Saved Q-table: {out / 'q_table.npy'}")
    print(f"Saved training reward history: {out_outputs / 'training_rewards.csv'}")
    print(f"Mean reward over last 100 episodes: {np.mean(rewards[-100:]):.3f}")


if __name__ == "__main__":
    main()

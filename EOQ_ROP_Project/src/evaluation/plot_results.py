"""Regenerate figures for the report/dashboard from saved experiment outputs."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rewards_path = OUT / "training_rewards.csv"
    if rewards_path.exists():
        rewards = pd.read_csv(rewards_path)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(rewards["episode"], rewards["total_reward"], alpha=0.25, label="Reward each episode")
        ax.plot(rewards["episode"], rewards["total_reward"].rolling(50, min_periods=1).mean(), label="Rolling mean (50)")
        ax.set_xlabel("Episode"); ax.set_ylabel("Simulated reward")
        ax.set_title("Q-learning training reward")
        ax.legend(); fig.tight_layout(); fig.savefig(FIG / "training_reward_curve.png", dpi=160); plt.close(fig)
    summary_path = OUT / "multi_seed_policy_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        mean_col = "simulated_total_reward_mean"
        std_col = "simulated_total_reward_std"
        if mean_col in summary:
            fig, ax = plt.subplots(figsize=(9, 4))
            x = summary["policy"].astype(str)
            y = summary[mean_col]
            yerr = summary[std_col] if std_col in summary else None
            ax.bar(x, y, yerr=yerr, capsize=3)
            ax.set_ylabel("Total simulated reward")
            ax.set_title("Policy comparison on held-out historical replay")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout(); fig.savefig(FIG / "policy_reward_comparison.png", dpi=160); plt.close(fig)
    print(f"Figures saved to {FIG}")


if __name__ == "__main__":
    main()

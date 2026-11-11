"""Project entry point: preprocess if needed, estimate elasticity and train Q-learning."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "processed" / "retail_price_clean.csv"
RAW = ROOT / "data" / "raw" / "retail_price.csv"


def main():
    if not CLEAN.exists():
        subprocess.run([sys.executable, str(ROOT / "src/data/preprocess.py"), "--input", str(RAW), "--output", str(CLEAN)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "src.rl.q_learning", "--data", str(CLEAN)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

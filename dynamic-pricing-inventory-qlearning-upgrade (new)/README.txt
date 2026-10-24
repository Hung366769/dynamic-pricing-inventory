Dynamic Pricing + Inventory notebook with added Q-learning section

Files:
- retail_price_elasticity_did_phase1_phase2_qlearning.ipynb
- retail_price.csv

Open the notebook in Jupyter/Colab. Keep retail_price.csv in the same working directory, then Run All.
Dependencies: pandas, numpy, matplotlib, scikit-learn.

The new Q-learning section is a simulation prototype based on the uploaded paper. It uses a training-only log-log-style demand model and explicit assumptions for cost and price guardrails. Simulated profits are not real-world causal estimates. Inventory-on-hand and lead time remain unavailable in the source dataset and are not fabricated.

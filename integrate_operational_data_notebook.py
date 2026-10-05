import json
from pathlib import Path


NOTEBOOK = Path("retail_price_elasticity_did_phase1_phase2_experiments.ipynb")


def lines(text: str) -> list[str]:
    text = text.strip("\n")
    return [line + "\n" for line in text.split("\n")]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


ops_intro = markdown(
    """
## 7.0 Operational data layer for IT-system realism

The original dataset contains observed price, demand and competitor signals.
For a fuller IT project workflow, the notebook now also loads generated
operational tables: product cost, inventory snapshots and purchase orders.
These tables make the pricing engine closer to a deployable decision-support
system while keeping the distinction clear: inventory and replenishment are
scenario data generated from reproducible rules, not raw observed inventory
records.
"""
)


ops_code = code(
    r"""
from pathlib import Path

OPERATIONAL_DIR = Path("operational_data")
required_operational_files = {
    "product_costs": OPERATIONAL_DIR / "product_costs.csv",
    "inventory_snapshots": OPERATIONAL_DIR / "inventory_snapshots.csv",
    "purchase_orders": OPERATIONAL_DIR / "purchase_orders.csv",
}
missing_operational = [str(path) for path in required_operational_files.values() if not path.exists()]
if missing_operational:
    raise FileNotFoundError(
        "Missing operational dataset(s). Run: py scripts/generate_operational_datasets.py"
    )

product_costs = pd.read_csv(required_operational_files["product_costs"])
inventory_snapshots = pd.read_csv(required_operational_files["inventory_snapshots"], parse_dates=["month_year"])
purchase_orders = pd.read_csv(
    required_operational_files["purchase_orders"],
    parse_dates=["order_date", "arrival_date"],
)

operational_features = [
    "product_id", "month_year", "opening_stock", "inventory_position",
    "reorder_point", "safety_stock", "lead_time_weeks", "sell_through_rate",
]
cost_features = [
    "product_id", "unit_cost", "cost_ratio_to_median_price",
    "holding_cost_per_unit_month", "stockout_penalty_per_unit",
]

operational_df = (
    real_df.merge(product_costs[cost_features], on="product_id", how="left")
    .merge(inventory_snapshots[operational_features], on=["product_id", "month_year"], how="left")
)

print("OPERATIONAL DATA CHECK")
print("Product costs:", product_costs.shape)
print("Inventory snapshots:", inventory_snapshots.shape)
print("Purchase orders:", purchase_orders.shape)
print("Operational modeling table:", operational_df.shape)
print("Missing operational values:", int(operational_df[cost_features[1:] + operational_features[2:]].isna().sum().sum()))
display(operational_df[
    ["product_id", "month_year", "qty", "unit_price", "unit_cost",
     "opening_stock", "reorder_point", "safety_stock", "lead_time_weeks"]
].head())
"""
)


replacement_ml_cell = r"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor


def add_operational_features(frame):
    f = frame.copy()
    f["month_year"] = pd.to_datetime(f["month_year"], dayfirst=True, errors="coerce")
    f["month"] = f["month_year"].dt.month.fillna(f.get("month", 1)).astype(int)
    f["year"] = f["month_year"].dt.year.fillna(f.get("year", 2017)).astype(int)
    competitor_cols = ["comp_1", "comp_2", "comp_3"]
    f["comp_avg"] = f[competitor_cols].mean(axis=1)
    f["comp_min"] = f[competitor_cols].min(axis=1)
    f["comp_max"] = f[competitor_cols].max(axis=1)
    f["price_gap_comp_avg"] = f["unit_price"] - f["comp_avg"]
    f["price_ratio_comp_avg"] = f["unit_price"] / f["comp_avg"].replace(0, np.nan)
    f["price_gap_lag"] = f["unit_price"] - f["lag_price"]
    f["price_ratio_lag"] = f["unit_price"] / f["lag_price"].replace(0, np.nan)
    f["freight_ratio_price"] = f["freight_price"] / f["unit_price"].replace(0, np.nan)
    if {"opening_stock", "reorder_point"}.issubset(f.columns):
        f["inventory_gap"] = f["opening_stock"] - f["reorder_point"]
        f["inventory_ratio_to_reorder"] = f["opening_stock"] / f["reorder_point"].replace(0, np.nan)
    if {"unit_cost", "unit_price"}.issubset(f.columns):
        f["gross_margin"] = f["unit_price"] - f["unit_cost"]
        f["gross_margin_pct"] = f["gross_margin"] / f["unit_price"].replace(0, np.nan)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


applied_source_df = operational_df if "operational_df" in globals() else real_df
applied_df = add_operational_features(applied_source_df)

CATEGORICAL_FEATURES = ["product_id", "product_category_name"]
NUMERIC_FEATURES = [
    "unit_price", "freight_price", "lag_price",
    "comp_1", "comp_2", "comp_3", "comp_avg", "comp_min", "comp_max",
    "price_gap_comp_avg", "price_ratio_comp_avg", "price_gap_lag",
    "price_ratio_lag", "freight_ratio_price",
    "product_name_lenght", "product_description_lenght", "product_photos_qty",
    "product_weight_g", "product_score", "customers",
    "weekday", "weekend", "holiday", "month", "year", "s", "volume",
    "unit_cost", "cost_ratio_to_median_price", "holding_cost_per_unit_month",
    "stockout_penalty_per_unit", "opening_stock", "inventory_position",
    "reorder_point", "safety_stock", "lead_time_weeks", "sell_through_rate",
    "inventory_gap", "inventory_ratio_to_reorder", "gross_margin", "gross_margin_pct",
]
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in applied_df.columns]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

model_df = applied_df.dropna(subset=FEATURE_COLUMNS + ["qty"]).copy()
train_parts, test_parts = [], []
for _, g in model_df.sort_values("month_year").groupby("product_id"):
    cut = int(np.floor(len(g) * 0.70))
    cut = min(max(cut, 1), len(g) - 1)
    train_parts.append(g.iloc[:cut])
    test_parts.append(g.iloc[cut:])

train_ml = pd.concat(train_parts, ignore_index=True)
test_ml = pd.concat(test_parts, ignore_index=True)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ],
    remainder="drop",
)

base_regressor = HistGradientBoostingRegressor(
    max_iter=350,
    learning_rate=0.045,
    max_leaf_nodes=15,
    l2_regularization=0.08,
    random_state=42,
)

demand_model = TransformedTargetRegressor(
    regressor=Pipeline([("prep", preprocess), ("model", base_regressor)]),
    func=np.log1p,
    inverse_func=np.expm1,
)

demand_model.fit(train_ml[FEATURE_COLUMNS], train_ml["qty"])
ml_pred = np.clip(demand_model.predict(test_ml[FEATURE_COLUMNS]), 0, None)

product_baseline = train_ml.groupby("product_id")["qty"].mean()
global_baseline = float(train_ml["qty"].mean())
baseline_pred = test_ml["product_id"].map(product_baseline).fillna(global_baseline).to_numpy()

ml_metrics = pd.DataFrame([
    {
        "model": "Applied ML demand model + ops data",
        "MAE": mean_absolute_error(test_ml["qty"], ml_pred),
        "RMSE": mean_squared_error(test_ml["qty"], ml_pred) ** 0.5,
        "R2": r2_score(test_ml["qty"], ml_pred),
    },
    {
        "model": "Product mean baseline",
        "MAE": mean_absolute_error(test_ml["qty"], baseline_pred),
        "RMSE": mean_squared_error(test_ml["qty"], baseline_pred) ** 0.5,
        "R2": r2_score(test_ml["qty"], baseline_pred),
    },
])

print(f"Applied ML train rows: {len(train_ml)} | test rows: {len(test_ml)}")
print(f"Feature count: {len(FEATURE_COLUMNS)}")
display(ml_metrics.round(3))

comparison = test_ml[["product_id", "product_category_name", "month_year", "qty", "unit_price"]].copy()
comparison["pred_qty_ml"] = ml_pred
comparison["baseline_qty"] = baseline_pred
display(comparison.head(12).round(2))
"""


replacement_pricing_cell = r"""
def make_price_scenarios(context_row, candidate_prices, competitor_multiplier=1.0):
    scenarios = pd.DataFrame([context_row.to_dict()] * len(candidate_prices))
    scenarios["unit_price"] = candidate_prices
    for col in ["comp_1", "comp_2", "comp_3"]:
        scenarios[col] = scenarios[col] * competitor_multiplier
    return add_operational_features(scenarios)


def elasticity_guardrail(category):
    row = real_price_results[real_price_results["category"] == category]
    if row.empty:
        return -1.0
    e = float(row.iloc[0]["elasticity"])
    if (not np.isfinite(e)) or e >= -0.15:
        e = -0.8
    return float(np.clip(e, -3.0, -0.30))


def get_product_unit_cost(context, observed_price, cost_frac=0.55):
    if "unit_cost" in context.index and pd.notna(context["unit_cost"]):
        return float(context["unit_cost"])
    return float(cost_frac * observed_price)


def forecast_price_grid(
    category,
    current_inventory=None,
    lead_time_weeks=None,
    safety_stock_weeks=1,
    competitor_multiplier=1.0,
    cost_frac=0.55,
    lo=0.70,
    hi=1.30,
    n_grid=121,
):
    g = model_df[model_df["product_category_name"] == category].sort_values("month_year")
    if g.empty:
        raise ValueError(f"Unknown category: {category}")

    context = g.iloc[-1].copy()
    observed_price = float(g["unit_price"].median())
    unit_cost = get_product_unit_cost(context, observed_price, cost_frac)
    if current_inventory is None:
        current_inventory = float(context.get("opening_stock", g["qty"].median() * 4))
    if lead_time_weeks is None:
        lead_time_weeks = float(context.get("lead_time_weeks", 2))

    min_price = max(unit_cost * 1.05, lo * observed_price)
    max_price = hi * observed_price
    candidate_prices = np.linspace(min_price, max_price, n_grid)

    baseline_scenario = make_price_scenarios(context, [observed_price], competitor_multiplier)
    baseline_demand = float(np.clip(demand_model.predict(baseline_scenario[FEATURE_COLUMNS])[0], 0, None))
    e = elasticity_guardrail(category)
    predicted_qty = baseline_demand * (candidate_prices / observed_price) ** e

    weekly_qty = g["qty"].astype(float) / 4.345
    weekly_mean = float(weekly_qty.mean())
    weekly_std = float(weekly_qty.std(ddof=1))
    if "reorder_point" in context.index and pd.notna(context["reorder_point"]):
        reorder_point = float(context["reorder_point"])
    else:
        safety_stock = safety_stock_weeks * weekly_mean
        reorder_point = weekly_mean * lead_time_weeks + 1.645 * weekly_std * np.sqrt(lead_time_weeks) + safety_stock

    if current_inventory < reorder_point:
        inventory_action = "REORDER / PROTECT STOCK"
    elif current_inventory > 1.30 * reorder_point:
        inventory_action = "HIGH STOCK / INCREASE SELL-THROUGH"
    else:
        inventory_action = "MAINTAIN"

    grid = pd.DataFrame({
        "candidate_price": candidate_prices,
        "predicted_qty": predicted_qty,
    })
    grid["unit_cost"] = unit_cost
    grid["baseline_demand_ml"] = baseline_demand
    grid["elasticity_guardrail"] = e
    grid["expected_revenue"] = grid["candidate_price"] * grid["predicted_qty"]
    grid["expected_profit"] = (grid["candidate_price"] - unit_cost) * grid["predicted_qty"]

    raw_inventory_pressure = (current_inventory - reorder_point) / max(reorder_point, 1)
    inventory_pressure = float(np.clip(raw_inventory_pressure, -2.0, 2.0))
    grid["inventory_adjusted_score"] = (
        grid["expected_profit"]
        + 0.60 * inventory_pressure * unit_cost * grid["predicted_qty"]
    )
    grid["reorder_point"] = reorder_point
    grid["current_inventory"] = current_inventory
    grid["inventory_action"] = inventory_action

    winner = grid.loc[grid["inventory_adjusted_score"].idxmax()].copy()
    return grid, winner


APPLIED_CATEGORY = "garden_tools"
latest_context = model_df[model_df["product_category_name"] == APPLIED_CATEGORY].sort_values("month_year").iloc[-1]
price_grid_ml, best_ml = forecast_price_grid(
    APPLIED_CATEGORY,
    current_inventory=float(latest_context.get("opening_stock", 120)),
    lead_time_weeks=float(latest_context.get("lead_time_weeks", 2)),
    safety_stock_weeks=1,
    competitor_multiplier=1.0,
)

print("Applied ML pricing recommendation")
print(f"Category: {APPLIED_CATEGORY}")
print(f"Recommended price: ${best_ml['candidate_price']:.2f}")
print(f"Predicted demand: {best_ml['predicted_qty']:.1f} units")
print(f"Expected profit: ${best_ml['expected_profit']:.2f}")
print(f"Inventory action: {best_ml['inventory_action']}")
print(f"Elasticity guardrail: {best_ml['elasticity_guardrail']:.2f}")
print(f"Reorder point: {best_ml['reorder_point']:.1f} | Current inventory: {best_ml['current_inventory']:.1f}")

display(
    price_grid_ml.sort_values("inventory_adjusted_score", ascending=False)
    .head(10)
    .round(2)
)

fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
ax1.plot(price_grid_ml["candidate_price"], price_grid_ml["expected_profit"], label="Expected profit")
ax1.plot(price_grid_ml["candidate_price"], price_grid_ml["inventory_adjusted_score"], label="Inventory-adjusted score")
ax1.axvline(best_ml["candidate_price"], ls="--", color="black", lw=1, label="Recommended price")
ax1.set_xlabel("Candidate price ($)")
ax1.set_ylabel("Model score")
ax1.set_title(f"Hybrid ML price optimization with operational inventory - {APPLIED_CATEGORY}")
ax1.legend()
fig.tight_layout()
plt.show()
"""


replacement_checklist = """
# 11. FINAL EXPERIMENT CHECKLIST FOR THE REPORT

- [x] Real `retail_price.csv` loaded and validated
- [x] Generated operational data layer: product costs, inventory snapshots and purchase orders
- [x] OLS/log-log elasticity kept as an interpretable benchmark, not the only decision model
- [x] Applied ML demand model trained with pricing, competitor, product, calendar and operational inventory/cost features
- [x] Out-of-time validation by product against a product-mean baseline
- [x] Business-readable feature importance for the ML demand model
- [x] Candidate-price simulation with a monotonic elasticity guardrail
- [x] Inventory-aware pricing score using current inventory, reorder point, lead time and safety stock
- [x] Competitor-price, inventory-level and lead-time scenarios for decision support
- [x] Combined dynamic pricing + inventory recommendation table
- [x] Reinforcement learning simulation for sequential pricing policy comparison

Important wording: price, demand and competitor signals come from the observed
retail pricing dataset. Cost, inventory, replenishment and lead-time tables are
generated operational scenario data to make the IT-system workflow realistic.
The updated decision model is a supervised ML demand-forecasting and hybrid
price-optimization prototype. Reinforcement learning is included as a
simulation extension for sequential decisions, not as direct causal evidence
from real market experimentation.
"""


with NOTEBOOK.open("r", encoding="utf-8") as f:
    nb = json.load(f)

insert_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "markdown" and "## 7.1 Experiment A" in "".join(cell.get("source", [])):
        insert_idx = i
        break
if insert_idx is None:
    raise RuntimeError("Could not find section 7.1")

already_has_ops = any(
    cell.get("cell_type") == "markdown"
    and "Operational data layer" in "".join(cell.get("source", []))
    for cell in nb["cells"]
)
if not already_has_ops:
    nb["cells"][insert_idx:insert_idx] = [ops_intro, ops_code]

for cell in nb["cells"]:
    if cell.get("cell_type") == "markdown":
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith("## Executive summary"):
            cell["source"] = lines(
                """
## Executive summary

This notebook supports a dynamic pricing and inventory decision workflow.
The early sections use log-log OLS elasticity and DiD as interpretable
benchmarks. The applied section upgrades the operational model to a supervised
ML demand forecaster, validates it out-of-time, and uses candidate price
simulation plus inventory pressure to recommend prices.

The project now includes an operational data layer with generated product cost,
inventory snapshot and purchase-order tables. This makes the notebook closer to
an IT decision-support system: raw pricing/demand data flows into operational
tables, feature engineering, demand forecasting, pricing optimization,
inventory actions and RL simulation.

A final reinforcement-learning section is included as a simulation extension:
it compares sequential pricing policies in a synthetic environment derived from
the demand model, elasticity guardrail and inventory rules. This should be
presented as scenario analysis rather than real-market causal proof.
"""
            )
        elif "FINAL EXPERIMENT CHECKLIST" in source:
            cell["source"] = lines(replacement_checklist)
    elif cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "Applied ML demand model" in source and "HistGradientBoostingRegressor" in source:
            cell["source"] = lines(replacement_ml_cell)
            cell["outputs"] = []
            cell["execution_count"] = None
        elif "def forecast_price_grid(" in source and "Applied ML pricing recommendation" in source:
            cell["source"] = lines(replacement_pricing_cell)
            cell["outputs"] = []
            cell["execution_count"] = None

with NOTEBOOK.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")


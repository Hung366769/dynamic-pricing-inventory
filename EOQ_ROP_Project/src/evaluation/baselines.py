"""Evaluate pricing and inventory policies in one historical-replay simulator.

All inventory/cost/lead-time quantities are scenario assumptions. Held-out
observed qty is used ex-post as the replay demand anchor, then price-adjusted
using the estimated elasticity. Results are counterfactual simulations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.environment.inventory_env import ACTIONS
from src.inventory_policies import calculate_eoq, calculate_rop, inventory_position


def _state(inv: float, inv_ref: float, forecast: float, median_qty: float,
           current_price: float, competitor_price: float, steps_left: int,
           horizon: int) -> tuple[int, int, int, int]:
    ratio_inv = inv / max(inv_ref, 1.0)
    inv_bin = 0 if ratio_inv < 0.33 else (2 if ratio_inv > 0.67 else 1)
    demand_ratio = forecast / max(median_qty, 1e-6)
    demand_bin = 0 if demand_ratio < 0.80 else (2 if demand_ratio > 1.20 else 1)
    price_ratio = current_price / max(competitor_price, 1e-6)
    price_bin = 0 if price_ratio < 0.97 else (2 if price_ratio > 1.03 else 1)
    rem = steps_left / max(horizon, 1)
    time_bin = 2 if rem > 0.66 else (1 if rem > 0.33 else 0)
    return inv_bin, demand_bin, price_bin, time_bin


def evaluate_policies(
    train_df: pd.DataFrame, test_df: pd.DataFrame, elasticity: float,
    q_table: np.ndarray | None = None, cost_ratio: float = 0.70,
    holding_cost_rate: float = 0.01, stockout_penalty_rate: float = 0.25,
    price_change_penalty_rate: float = 0.05, replenishment_rate: float = 0.0,
    max_price_deviation: float = 0.30, initial_stock_cover: float = 2.5,
    ordering_cost: float = 25.0, annual_holding_cost_rate: float = 0.20,
    lead_time_periods: int = 1, safety_stock_units: float = 0.0,
    fixed_order_quantity: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_df.copy(), test_df.copy()
    for frame in (train, test):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "competitor_price" not in frame.columns:
            comp = [c for c in ["comp_1", "comp_2", "comp_3"] if c in frame.columns]
            frame["competitor_price"] = frame[comp].replace(0, np.nan).mean(axis=1)
    required = ["date", "product_id", "qty", "unit_price", "competitor_price"]
    train = train.dropna(subset=required)
    test = test.dropna(subset=required)
    products = sorted(set(test["product_id"]) & set(train["product_id"]))
    policy_names = [
        "fixed_price_no_replenishment", "fixed_qty_rop", "eoq_rop",
        "random_policy", "match_competitor", "inventory_rule"
    ]
    if q_table is not None:
        policy_names.append("q_learning")
    random_rng = np.random.default_rng(2026)

    train_stats, train_history = {}, {}
    for pid, g in train.groupby("product_id"):
        g = g.sort_values("date").reset_index(drop=True)
        train_stats[pid] = {
            "median_qty": max(float(g["qty"].median()), 1.0),
            "median_price": float(g["unit_price"].median()),
            "last_price": float(g.iloc[-1]["unit_price"]),
        }
        train_history[pid] = g

    details = []
    for pid in products:
        hist_train = train_history[pid]
        product_test = test[test["product_id"] == pid].sort_values("date").reset_index(drop=True)
        if product_test.empty:
            continue
        st = train_stats[pid]
        horizon, median_qty = len(product_test), st["median_qty"]
        initial_inventory = max(1, int(round(median_qty * initial_stock_cover)))
        inventory_reference = max(initial_inventory, median_qty * horizon)
        capacity = max(inventory_reference * 2, initial_inventory)
        unit_cost = cost_ratio * st["median_price"]
        price_anchor = st["median_price"]
        lower, upper = price_anchor * (1-max_price_deviation), price_anchor * (1+max_price_deviation)
        initial_price = float(np.clip(st["last_price"], lower, upper))

        # Monthly replay periods: D is annualized from training median monthly demand.
        annual_demand = median_qty * 12.0
        annual_holding_cost_per_unit = max(unit_cost * annual_holding_cost_rate, 1e-9)
        eoq_qty = calculate_eoq(annual_demand, ordering_cost, annual_holding_cost_per_unit)
        fixed_qty = int(fixed_order_quantity or max(1, round(median_qty * 2.0)))
        rop = calculate_rop(median_qty, lead_time_periods, safety_stock_units)

        for policy in policy_names:
            inventory, current_price = initial_inventory, initial_price
            history_qty = hist_train["qty"].astype(float).tolist()
            # Queue contains (arrival_index, quantity); inventory position counts all open orders.
            open_orders: list[tuple[int, int]] = []
            total_ordering_cost = total_holding_cost = 0.0
            order_count = 0
            for i, row in product_test.iterrows():
                arriving = sum(q for arrival, q in open_orders if arrival <= i)
                open_orders = [(arrival, q) for arrival, q in open_orders if arrival > i]
                inventory = min(capacity, inventory + arriving)

                forecast_vals = history_qty[-3:]
                forecast = float(np.mean(forecast_vals)) if forecast_vals else median_qty
                competitor_price = float(row["competitor_price"])
                state = _state(inventory, inventory_reference, forecast, median_qty,
                               current_price, competitor_price, horizon-i, horizon)
                if policy == "fixed_price_no_replenishment":
                    action_index = 2
                    price = initial_price
                elif policy == "q_learning" and q_table is not None:
                    action_index = int(np.argmax(q_table[state]))
                    price = float(np.clip(current_price*(1+ACTIONS[action_index]), lower, upper))
                elif policy == "random_policy":
                    action_index = int(random_rng.integers(len(ACTIONS)))
                    price = float(np.clip(current_price*(1+ACTIONS[action_index]), lower, upper))
                elif policy == "match_competitor":
                    candidates = [float(np.clip(current_price*(1+a), lower, upper)) for a in ACTIONS]
                    action_index = int(np.argmin([abs(x-competitor_price) for x in candidates]))
                    price = candidates[action_index]
                elif policy == "inventory_rule":
                    ratio = inventory / max(inventory_reference, 1.0)
                    action_index = 0 if ratio > .67 else (3 if ratio < .33 else 2)
                    price = float(np.clip(current_price*(1+ACTIONS[action_index]), lower, upper))
                else:  # Same Q-learning price actions, with alternate replenishment rules.
                    action_index = 2
                    price = current_price

                base_demand = float(row["qty"])
                expected_demand = base_demand * max(price/float(row["unit_price"]), 1e-6)**elasticity
                demand = max(0, int(round(expected_demand)))
                sold = min(demand, int(inventory))
                unmet = max(demand-sold, 0)
                inventory = max(0, int(inventory)-sold)

                order_qty = 0
                if policy in ("fixed_qty_rop", "eoq_rop", "q_learning"):
                    on_order = sum(q for _, q in open_orders)
                    position = inventory_position(inventory, on_order)
                    if position <= rop:
                        order_qty = fixed_qty if policy == "fixed_qty_rop" else eoq_qty
                        if order_qty > 0:
                            arrival = i + max(1, int(lead_time_periods))
                            open_orders.append((arrival, order_qty))
                            order_count += 1
                            total_ordering_cost += ordering_cost
                elif policy in ("random_policy", "match_competitor", "inventory_rule"):
                    # These legacy pricing-only baselines retain the configurable
                    # fixed replenishment-rate behavior; q_learning uses EOQ + ROP.
                    legacy_add = max(0, int(round(replenishment_rate * median_qty)))
                    inventory = min(capacity, inventory + legacy_add)

                gross_profit = (price-unit_cost)*sold
                holding_cost = holding_cost_rate*unit_cost*inventory
                total_holding_cost += holding_cost
                stockout_cost = stockout_penalty_rate*unit_cost*unmet
                price_change_cost = (price_change_penalty_rate*unit_cost*max(median_qty,1.0)
                                     * abs(price-current_price)/max(float(row["unit_price"]),1e-6))
                reward = gross_profit - holding_cost - stockout_cost - price_change_cost - (
                    ordering_cost if order_qty else 0.0
                )
                details.append({
                    "policy": policy, "product_id": pid, "date": row["date"],
                    "price_simulated": price, "reference_price_observed": float(row["unit_price"]),
                    "competitor_price": competitor_price, "demand_simulated": demand,
                    "quantity_sold_simulated": sold, "unmet_demand_simulated": unmet,
                    "inventory_begin": inventory + sold, "inventory_end": inventory,
                    "inventory_position_end": inventory + sum(q for _,q in open_orders),
                    "replenishment_order_qty": order_qty, "eoq_quantity": eoq_qty,
                    "reorder_point": rop, "orders_placed_cumulative": order_count,
                    "revenue_simulated": price*sold, "gross_profit_simulated": gross_profit,
                    "ordering_cost_simulated": ordering_cost if order_qty else 0.0,
                    "holding_cost_simulated": holding_cost, "stockout_cost_simulated": stockout_cost,
                    "price_change_cost_simulated": price_change_cost, "reward_simulated": reward,
                })
                current_price = price
                history_qty.append(float(row["qty"]))

    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        return detail_df, pd.DataFrame()
    summaries = []
    for policy, g in detail_df.groupby("policy", sort=False):
        total_demand = float(g["demand_simulated"].sum())
        summaries.append({
            "policy": policy,
            "simulated_revenue": float(g["revenue_simulated"].sum()),
            "simulated_gross_profit": float(g["gross_profit_simulated"].sum()),
            "simulated_total_reward": float(g["reward_simulated"].sum()),
            "simulated_units_sold": int(g["quantity_sold_simulated"].sum()),
            "simulated_ending_inventory": int(g.groupby("product_id").tail(1)["inventory_end"].sum()),
            "simulated_unmet_demand": int(g["unmet_demand_simulated"].sum()),
            "simulated_stockout_rate_pct": float(g["unmet_demand_simulated"].sum()/max(total_demand,1e-12)*100),
            "simulated_ordering_cost": float(g["ordering_cost_simulated"].sum()),
            "simulated_holding_cost": float(g["holding_cost_simulated"].sum()),
            "simulated_total_inventory_cost": float(g["ordering_cost_simulated"].sum()+g["holding_cost_simulated"].sum()),
            "simulated_orders_placed": int(g.groupby("product_id")["orders_placed_cumulative"].max().sum()),
            "mean_eoq_units": float(g["eoq_quantity"].mean()),
            "mean_reorder_point_units": float(g["reorder_point"].mean()),
            "simulated_price_change_abs_mean": float(g.groupby("product_id")["price_simulated"].diff().abs().mean()),
        })
    summary_df = pd.DataFrame(summaries).sort_values("simulated_total_reward", ascending=False).reset_index(drop=True)
    return detail_df, summary_df

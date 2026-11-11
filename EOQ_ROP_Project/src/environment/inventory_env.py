"""A transparent historical-replay simulator for pricing and inventory.

Important: this is a simulator, not an actual inventory record. At each step,
observed qty is treated as the demand baseline at the observed unit_price; the
counterfactual demand at a new price is scaled using the estimated elasticity.
Initial inventory, unit cost, replenishment, holding cost and stockout penalty
are user-specified assumptions and must be sensitivity-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inventory_policies import (
    calculate_eoq,
    calculate_rop,
    inventory_position
)

ACTIONS = np.array([-0.10, -0.05, 0.00, 0.05, 0.10], dtype=float)
STATE_DIMS = (3, 3, 3, 3)  # inventory, demand forecast, price-vs-competitor, time


class InventoryPricingEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        elasticity: float,
        seed: int = 42,
        episode_len: int = 6,
        cost_ratio: float = 0.70,
        holding_cost_rate: float = 0.01,
        stockout_penalty_rate: float = 0.25,
        price_change_penalty_rate: float = 0.05,
        replenishment_rate: float = 0.0,
        max_price_deviation: float = 0.30,
        noise_sigma: float = 0.10,
        inventory_policy: str = "eoq_rop",
        ordering_cost: float = 25.0,
        annual_holding_cost_rate: float = 0.20,
        lead_time_periods: int = 1,
        safety_stock_units: float = 0.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.elasticity = float(elasticity)
        self.episode_len = int(episode_len)
        self.cost_ratio = float(cost_ratio)
        self.holding_cost_rate = float(holding_cost_rate)
        self.stockout_penalty_rate = float(stockout_penalty_rate)
        self.price_change_penalty_rate = float(price_change_penalty_rate)
        self.replenishment_rate = float(replenishment_rate)
        self.max_price_deviation = float(max_price_deviation)
        self.noise_sigma = float(noise_sigma)
        self.inventory_policy = str(inventory_policy)
        self.ordering_cost = float(ordering_cost)
        self.annual_holding_cost_rate = float(annual_holding_cost_rate)
        self.lead_time_periods = max(1, int(lead_time_periods))
        self.safety_stock_units = max(0.0, float(safety_stock_units))
        from src.inventory_policies import calculate_eoq, calculate_rop

        self.data = data.copy()
        if "date" in self.data.columns:
            self.data["date"] = pd.to_datetime(self.data["date"], errors="coerce")
        else:
            self.data["date"] = pd.to_datetime(self.data["month_year"], format="%d-%m-%Y", errors="coerce")
        if "competitor_price" not in self.data.columns:
            comp_cols = [c for c in ["comp_1", "comp_2", "comp_3"] if c in self.data.columns]
            self.data["competitor_price"] = self.data[comp_cols].replace(0, np.nan).mean(axis=1)
        self.data = self.data.dropna(subset=["date", "product_id", "qty", "unit_price", "competitor_price"])
        self.data = self.data[(self.data["qty"] > 0) & (self.data["unit_price"] > 0) & (self.data["competitor_price"] > 0)]
        self.data = self.data.sort_values(["product_id", "date"]).reset_index(drop=True)
        self.groups = {}
        self.product_stats = {}
        for pid, group in self.data.groupby("product_id", sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) >= 4:
                self.groups[pid] = group
                self.product_stats[pid] = {
                    "median_qty": float(group["qty"].median()),
                    "mean_qty": float(group["qty"].mean()),
                    "median_price": float(group["unit_price"].median()),
                }
        if not self.groups:
            raise ValueError("Need at least one product with four or more observations.")
        self.product_ids = list(self.groups)
        self.current_product = None

    @staticmethod
    def _bucket_3(value: float, low: float, high: float) -> int:
        if value < low:
            return 0
        if value > high:
            return 2
        return 1

    def _state(self) -> tuple[int, int, int, int]:
        row = self.rows.loc[self.idx]
        med_qty = max(self.product_stats[self.current_product]["median_qty"], 1e-6)
        inv_ratio = self.inventory / max(self.inventory_reference, 1.0)
        inv_bin = self._bucket_3(inv_ratio, 0.33, 0.67)

        hist_start = max(0, self.idx - 3)
        historical_qty = self.rows.loc[hist_start : self.idx - 1, "qty"].astype(float).tolist()
        forecast = float(np.mean(historical_qty)) if historical_qty else med_qty
        demand_ratio = forecast / med_qty
        demand_bin = self._bucket_3(demand_ratio, 0.80, 1.20)

        comp = max(float(row["competitor_price"]), 1e-6)
        price_ratio = float(self.current_price) / comp
        price_bin = self._bucket_3(price_ratio, 0.97, 1.03)

        remaining = max(self.episode_steps - self.steps_taken, 0) / max(self.episode_steps, 1)
        time_bin = 2 if remaining > 0.66 else (1 if remaining > 0.33 else 0)
        return (inv_bin, demand_bin, price_bin, time_bin)

    def reset(self) -> tuple[tuple[int, int, int, int], dict]:
        self.current_product = str(self.rng.choice(self.product_ids))
        self.rows = self.groups[self.current_product].reset_index(drop=True)
        n = len(self.rows)
        self.episode_steps = min(self.episode_len, n - 1)
        # Start after at least one historical observation so the agent has a
        # demand-forecast proxy using only prior rows.
        max_start = max(1, n - self.episode_steps)
        self.start_idx = int(self.rng.integers(1, max_start + 1))
        self.idx = self.start_idx
        self.steps_taken = 0
        med_qty = max(self.product_stats[self.current_product]["median_qty"], 1.0)
        coverage = float(self.rng.uniform(0.5, 1.5))
        self.inventory = max(1, int(round(med_qty * self.episode_steps * coverage)))
        self.inventory_reference = max(self.inventory, med_qty * self.episode_steps)
        self.capacity = max(self.inventory_reference * 2, self.inventory)
        self.unit_cost = self.cost_ratio * self.product_stats[self.current_product]["median_price"]
        annual_demand = med_qty * 12.0
        annual_holding = max(self.unit_cost * self.annual_holding_cost_rate, 1e-9)
        self.eoq_quantity = calculate_eoq(annual_demand, self.ordering_cost, annual_holding)
        self.reorder_point = calculate_rop(med_qty, self.lead_time_periods, self.safety_stock_units)
        self.open_orders = []
        self.current_price = float(self.rows.loc[self.idx, "unit_price"])
        self.total_reward = 0.0
        self.history = []
        return self._state(), {"product_id": self.current_product, "start_idx": self.start_idx}

    def step(self, action_index: int):
        if action_index < 0 or action_index >= len(ACTIONS):
            raise ValueError(f"action_index must be 0..{len(ACTIONS)-1}")
        row = self.rows.loc[self.idx]
        reference_price = float(row["unit_price"])
        old_price = float(self.current_price)
        candidate = old_price * (1.0 + float(ACTIONS[action_index]))
        # Stable price guardrails are relative to the product's training-period
        # median price, not the current-month observed price (which is only a
        # replay reference for demand simulation).
        product_price_anchor = self.product_stats[self.current_product]["median_price"]
        lower = product_price_anchor * (1.0 - self.max_price_deviation)
        upper = product_price_anchor * (1.0 + self.max_price_deviation)
        price = float(np.clip(candidate, lower, upper))

        # Scenario assumption: observed qty anchors demand at the observed price.
        # RL does not observe current row qty before choosing the action.
        base_demand = float(row["qty"])
        expected_demand = base_demand * max(price / reference_price, 1e-6) ** self.elasticity
        noise = float(self.rng.normal(-0.5 * self.noise_sigma**2, self.noise_sigma))
        demand = max(0, int(round(expected_demand * np.exp(noise))))
        sold = min(demand, int(self.inventory))
        unmet = max(demand - sold, 0)
        leftover = max(int(self.inventory) - sold, 0)
        median_qty = max(self.product_stats[self.current_product]["median_qty"], 1.0)
        next_inventory = int(leftover)
        # Receive orders due at the beginning of the next simulated period.
        arriving = sum(q for arrival_step, q in self.open_orders if arrival_step <= self.steps_taken + 1)
        self.open_orders = [(a, q) for a, q in self.open_orders if a > self.steps_taken + 1]
        next_inventory = min(int(self.capacity), next_inventory + arriving)
        order_qty = 0
        ordering_cost = 0.0
        if self.inventory_policy == "eoq_rop":
            on_order = sum(q for _, q in self.open_orders)
            if next_inventory + on_order <= self.reorder_point and self.eoq_quantity > 0:
                order_qty = self.eoq_quantity
                self.open_orders.append((self.steps_taken + self.lead_time_periods, order_qty))
                ordering_cost = self.ordering_cost
        elif self.inventory_policy == "fixed_rate":
            replenishment = max(0, int(round(self.replenishment_rate * median_qty)))
            next_inventory = min(int(self.capacity), next_inventory + replenishment)

        gross_profit = (price - self.unit_cost) * sold
        holding_cost = self.holding_cost_rate * self.unit_cost * next_inventory
        stockout_cost = self.stockout_penalty_rate * self.unit_cost * unmet
        price_change_cost = (
            self.price_change_penalty_rate
            * self.unit_cost
            * max(median_qty, 1.0)
            * abs(price - old_price) / max(reference_price, 1e-6)
        )
        reward = float(gross_profit - holding_cost - stockout_cost - price_change_cost - ordering_cost)

        info = {
            "product_id": self.current_product,
            "date": str(row["date"].date()),
            "price": price,
            "observed_reference_price": reference_price,
            "demand_simulated": demand,
            "quantity_sold": sold,
            "unmet_demand": unmet,
            "inventory_end": next_inventory,
            "gross_profit_simulated": gross_profit,
            "holding_cost_simulated": holding_cost,
            "ordering_cost_simulated": ordering_cost,
            "replenishment_order_qty": order_qty,
            "eoq_quantity": self.eoq_quantity,
            "reorder_point": self.reorder_point,
            "inventory_position_end": next_inventory + sum(q for _, q in self.open_orders),
            "stockout_cost_simulated": stockout_cost,
            "reward": reward,
        }
        self.history.append(info)
        self.total_reward += reward
        self.inventory = next_inventory
        self.current_price = price
        self.idx += 1
        self.steps_taken += 1
        done = self.steps_taken >= self.episode_steps or self.idx >= len(self.rows)
        next_state = None if done else self._state()
        return next_state, reward, done, info

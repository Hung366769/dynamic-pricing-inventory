from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "retail_price.csv"
OUT_DIR = ROOT / "operational_data"


def stable_code(value: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(value))


def generate_product_costs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    category_cost_shift = {
        "bed_bath_table": 0.56,
        "computers_accessories": 0.62,
        "consoles_games": 0.64,
        "cool_stuff": 0.54,
        "furniture_decor": 0.58,
        "garden_tools": 0.55,
        "health_beauty": 0.48,
        "perfumery": 0.46,
        "watches_gifts": 0.57,
    }
    for product_id, g in df.groupby("product_id"):
        category = g["product_category_name"].iloc[0]
        base_price = float(g["unit_price"].median())
        code = stable_code(product_id)
        cost_ratio = category_cost_shift.get(category, 0.55) + ((code % 9) - 4) * 0.008
        cost_ratio = float(np.clip(cost_ratio, 0.42, 0.72))
        unit_cost = round(base_price * cost_ratio, 2)
        rows.append(
            {
                "product_id": product_id,
                "product_category_name": category,
                "unit_cost": unit_cost,
                "cost_ratio_to_median_price": round(cost_ratio, 3),
                "supplier_id": f"SUP-{(code % 8) + 1:02d}",
                "min_order_qty": int(max(10, round(g["qty"].mean() * 2))),
                "holding_cost_per_unit_month": round(unit_cost * 0.018, 2),
                "stockout_penalty_per_unit": round(unit_cost * 0.35, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("product_id").reset_index(drop=True)


def generate_inventory_and_purchase_orders(df: pd.DataFrame, costs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost_lookup = costs.set_index("product_id").to_dict("index")
    inventory_rows = []
    po_rows = []
    po_id = 1

    for product_id, g in df.sort_values("month_year").groupby("product_id"):
        g = g.copy()
        category = g["product_category_name"].iloc[0]
        stats_mean = float(g["qty"].mean())
        stats_std = float(g["qty"].std(ddof=1)) if len(g) > 1 else max(stats_mean * 0.25, 1.0)
        code = stable_code(product_id)
        lead_time_weeks = int(1 + code % 5)
        lead_time_months = lead_time_weeks / 4.345
        safety_stock = int(np.ceil(0.50 * stats_mean + 0.65 * stats_std))
        reorder_point = int(np.ceil(stats_mean * lead_time_months + safety_stock))
        min_order_qty = int(cost_lookup[product_id]["min_order_qty"])
        target_stock = int(np.ceil(max(reorder_point + stats_mean * 1.8, stats_mean * 2.5, min_order_qty)))
        stock = int(np.ceil(target_stock + stats_std))

        for _, row in g.iterrows():
            observed_demand = int(row["qty"])
            opening_stock = int(stock)
            fulfilled_qty = int(min(opening_stock, observed_demand))
            lost_sales = int(max(observed_demand - opening_stock, 0))
            ending_stock = int(max(opening_stock - observed_demand, 0))
            stockout_flag = int(lost_sales > 0)
            inventory_position = ending_stock
            sell_through_rate = fulfilled_qty / max(opening_stock, 1)

            if inventory_position <= reorder_point:
                order_qty = int(max(target_stock - inventory_position, min_order_qty))
                order_date = row["month_year"] + pd.Timedelta(days=3)
                arrival_date = order_date + pd.Timedelta(days=lead_time_weeks * 7)
                po_rows.append(
                    {
                        "purchase_order_id": f"PO-{po_id:05d}",
                        "product_id": product_id,
                        "product_category_name": category,
                        "order_date": order_date.date().isoformat(),
                        "arrival_date": arrival_date.date().isoformat(),
                        "quantity_ordered": order_qty,
                        "supplier_id": cost_lookup[product_id]["supplier_id"],
                        "lead_time_weeks": lead_time_weeks,
                        "unit_cost": cost_lookup[product_id]["unit_cost"],
                    }
                )
                po_id += 1
                ending_stock += order_qty
                inventory_position = ending_stock

            inventory_rows.append(
                {
                    "product_id": product_id,
                    "product_category_name": category,
                    "month_year": row["month_year"].date().isoformat(),
                    "observed_demand": observed_demand,
                    "opening_stock": opening_stock,
                    "fulfilled_qty": fulfilled_qty,
                    "lost_sales": lost_sales,
                    "ending_stock": ending_stock,
                    "inventory_position": inventory_position,
                    "reorder_point": reorder_point,
                    "safety_stock": safety_stock,
                    "lead_time_weeks": lead_time_weeks,
                    "stockout_flag": stockout_flag,
                    "sell_through_rate": round(sell_through_rate, 4),
                }
            )
            stock = ending_stock

    inventory = pd.DataFrame(inventory_rows).sort_values(["product_id", "month_year"]).reset_index(drop=True)
    purchase_orders = pd.DataFrame(po_rows).sort_values(["order_date", "product_id"]).reset_index(drop=True)
    return inventory, purchase_orders


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(SOURCE, parse_dates=["month_year"], dayfirst=True)
    costs = generate_product_costs(df)
    inventory, purchase_orders = generate_inventory_and_purchase_orders(df, costs)

    costs.to_csv(OUT_DIR / "product_costs.csv", index=False)
    inventory.to_csv(OUT_DIR / "inventory_snapshots.csv", index=False)
    purchase_orders.to_csv(OUT_DIR / "purchase_orders.csv", index=False)

    print(f"Wrote {len(costs)} product costs")
    print(f"Wrote {len(inventory)} inventory snapshots")
    print(f"Wrote {len(purchase_orders)} purchase orders")


if __name__ == "__main__":
    main()

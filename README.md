# dynamic-pricing-inventory

Dynamic pricing and inventory management using an applied ML demand
forecasting model, hybrid candidate-price simulation, competitor signals, and
inventory-aware reorder/sell-through decisions.

The notebook keeps log-log OLS elasticity as an interpretable benchmark, but
the operational recommendation layer uses supervised demand forecasting plus a
monotonic elasticity guardrail for price-grid optimization.

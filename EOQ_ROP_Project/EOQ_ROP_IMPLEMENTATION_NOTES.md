# EOQ + ROP implementation notes

## What changed
- Added `src/inventory_policies.py` with EOQ, ROP and inventory-position calculations.
- Added simulated EOQ + ROP replenishment with an outstanding-order queue and lead time to the Q-learning environment.
- Added two inventory evaluation policies: `fixed_qty_rop` and `eoq_rop`.
- Added ordering-cost, holding-cost, lead-time and safety-stock parameters.
- Added `run_eoq_sensitivity.py` for scenario sensitivity tests.

## Assumptions (not observed in the dataset)
Defaults are illustrative only: ordering cost = 25; annual holding-cost rate = 20% of assumed unit cost; lead time = 1 month; safety stock = 0. Annual demand is estimated as 12 × median monthly demand in the training subset. Fixed order quantity is 2 × median monthly demand.

## Interpretation
The EOQ + ROP policies in the holdout evaluation keep the price path fixed to isolate the inventory policy effect. Q-learning is also evaluated as a pricing policy. In training and holdout evaluation, Q-learning interacts with EOQ + ROP replenishment. All reported profits, rewards, stockouts and costs remain simulation outputs.

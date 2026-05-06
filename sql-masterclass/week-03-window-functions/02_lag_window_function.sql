-- LAG() window function — same result, one table scan
-- Cost on BigQuery: ~0.8 GB scanned per run
-- At 19 dashboard refreshes/day = $0.08/day, $2.40/month
-- Monthly saving vs self-join: $347

SELECT
  user_id,
  order_date,
  order_value,
  LAG(order_value) OVER (
    PARTITION BY user_id
    ORDER BY order_date
  ) AS prev_order_value,

  -- Bonus: delta calculation inline, no extra join
  order_value - LAG(order_value) OVER (
    PARTITION BY user_id
    ORDER BY order_date
  ) AS order_value_delta

FROM orders
ORDER BY user_id, order_date;

-- BigQuery: 0.8 GB scanned = $0.004/run × 19/day = $0.08/day
-- How it works:
-- PARTITION BY user_id  → separate window per user
-- ORDER BY order_date   → defines "previous" row
-- LAG(col, 1, NULL)     → offset=1 (default), default_value=NULL if no prior row

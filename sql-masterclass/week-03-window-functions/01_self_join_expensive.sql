-- Self-join pattern to get previous row per user
-- Cost on BigQuery: ~126 GB scanned per run
-- At 19 dashboard refreshes/day = $12/day, $347/month

SELECT
  a.user_id,
  a.order_date,
  a.order_value,
  b.order_value AS prev_order_value
FROM orders a
LEFT JOIN orders b
  ON a.user_id = b.user_id
  AND b.order_date = (
    SELECT MAX(order_date) FROM orders c
    WHERE c.user_id = a.user_id
      AND c.order_date < a.order_date
  )
ORDER BY a.user_id, a.order_date;

-- Why it's expensive:
-- 1. Correlated subquery fires once per row = O(n) subquery calls
-- 2. Self-join reads the full table twice
-- 3. Query planner cannot merge both passes into one scan
-- 4. On 100M rows: 126 GB scanned = $0.63/run
-- 5. 19 refreshes/day = $11.97/day = $359/month

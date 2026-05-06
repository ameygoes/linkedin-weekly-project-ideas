-- More window functions — replace expensive patterns with single-pass SQL

-- ─────────────────────────────────────────────────────
-- 1. LEAD() — peek at the NEXT row (churn signal)
-- ─────────────────────────────────────────────────────
SELECT
  user_id,
  order_date,
  order_value,
  LEAD(order_date) OVER (
    PARTITION BY user_id
    ORDER BY order_date
  ) AS next_order_date,
  DATE_DIFF(
    LEAD(order_date) OVER (PARTITION BY user_id ORDER BY order_date),
    order_date,
    DAY
  ) AS days_until_next_order
FROM orders;
-- If days_until_next_order IS NULL → this was the last order (churn candidate)

-- ─────────────────────────────────────────────────────
-- 2. ROW_NUMBER() — deduplicate without a subquery
-- ─────────────────────────────────────────────────────
-- Keep only the most recent order per user
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY order_date DESC
    ) AS rn
  FROM orders
)
SELECT * EXCEPT(rn)
FROM ranked
WHERE rn = 1;

-- ─────────────────────────────────────────────────────
-- 3. Running total with SUM() OVER — no GROUP BY needed
-- ─────────────────────────────────────────────────────
SELECT
  order_date,
  order_value,
  SUM(order_value) OVER (
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_total,
  SUM(order_value) OVER (
    PARTITION BY DATE_TRUNC(order_date, MONTH)
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS monthly_running_total
FROM orders;

-- ─────────────────────────────────────────────────────
-- 4. NTILE() — quartile bucketing without a subquery
-- ─────────────────────────────────────────────────────
SELECT
  user_id,
  SUM(order_value) AS total_revenue,
  NTILE(4) OVER (ORDER BY SUM(order_value) DESC) AS revenue_quartile
  -- 1 = top 25%, 4 = bottom 25%
FROM orders
GROUP BY user_id;

-- ─────────────────────────────────────────────────────
-- 5. RANK() vs DENSE_RANK() — know the difference
-- ─────────────────────────────────────────────────────
SELECT
  user_id,
  order_value,
  RANK() OVER (ORDER BY order_value DESC) AS rank_with_gaps,
  -- 1, 2, 2, 4 (skips 3 when there's a tie)
  DENSE_RANK() OVER (ORDER BY order_value DESC) AS rank_no_gaps
  -- 1, 2, 2, 3 (no skips)
FROM orders;

-- ─────────────────────────────────────────────────────
-- 6. Moving average — 7-day rolling window
-- ─────────────────────────────────────────────────────
SELECT
  order_date,
  order_value,
  AVG(order_value) OVER (
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS avg_last_7_days
FROM orders;

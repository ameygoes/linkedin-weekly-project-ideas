# SQL Masterclass — Window Functions

> Self-join on 100M rows: **$12/day** on BigQuery.  
> One window function: **$0.08/day**.  
> $347/month saved. Same result.

## The Problem

The classic "get previous row per user" query uses a self-join with a correlated subquery:

```sql
SELECT a.*, b.order_value AS prev_order_value
FROM orders a
LEFT JOIN orders b
  ON a.user_id = b.user_id
  AND b.order_date = (
    SELECT MAX(order_date) FROM orders c
    WHERE c.user_id = a.user_id AND c.order_date < a.order_date
  )
```

On 100M rows: **126 GB scanned = $0.63/run**. At 19 dashboard refreshes/day → **$12/day**.

## The Fix

```sql
SELECT
  user_id, order_date, order_value,
  LAG(order_value) OVER (
    PARTITION BY user_id
    ORDER BY order_date
  ) AS prev_order_value
FROM orders
```

One table scan. **0.8 GB scanned = $0.004/run**. 19 refreshes/day → **$0.08/day**.

## Files

| File | What it shows |
|------|--------------|
| `01_self_join_expensive.sql` | The slow pattern — correlated subquery + self-join |
| `02_lag_window_function.sql` | LAG() fix with delta calculation inline |
| `03_more_window_functions.sql` | LEAD, ROW_NUMBER, SUM running total, NTILE, RANK, moving average |

## Why Window Functions Win

- **One pass**: the engine scans the table once, computes all window results in memory
- **No correlated subqueries**: no O(n²) hidden in readable-looking SQL
- **Composable**: stack multiple OVER() clauses in one SELECT — no extra joins or CTEs

## Cost Breakdown (BigQuery, $6.25/TB)

| Approach | Scan per run | Cost/run | Cost/day (19×) | Cost/month |
|----------|-------------|----------|----------------|------------|
| Self-join + correlated subquery | 126 GB | $0.63 | $11.97 | $359 |
| LAG() window function | 0.8 GB | $0.004 | $0.076 | $2.28 |
| **Saving** | | | **$11.89/day** | **$357/month** |

## Window Function Syntax

```sql
function_name(expression) OVER (
  PARTITION BY col    -- separate window per group (optional)
  ORDER BY col        -- defines row order within the window
  ROWS BETWEEN ...    -- frame clause (optional)
)
```

Common frame clauses:
- `ROWS UNBOUNDED PRECEDING AND CURRENT ROW` — running total
- `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` — 7-day rolling window
- `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` — full partition aggregate

## Functions Covered

| Function | Use case |
|----------|---------|
| `LAG(col, n)` | Previous nth row value |
| `LEAD(col, n)` | Next nth row value |
| `ROW_NUMBER()` | Unique rank per partition (dedup) |
| `RANK()` | Rank with gaps on ties |
| `DENSE_RANK()` | Rank without gaps on ties |
| `NTILE(n)` | Divide into n equal buckets |
| `SUM() OVER` | Running/moving total |
| `AVG() OVER` | Moving average |

---

Part of the [@AccidentalDataEngineer](https://linkedin.com/in/accidentaldataengineer) SQL Masterclass series.

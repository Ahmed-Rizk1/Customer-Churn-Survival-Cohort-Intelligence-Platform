-- ============================================================================
-- Project 4: SaaS Churn Survival & Cohort Intelligence Suite
-- File: 02_cohort_retention_matrix.sql
-- Purpose: Triangular Cohort Retention Matrix evaluating monthly subscriber decay
-- Techniques: Recursive CTEs, Window Functions (MIN() OVER PARTITION BY), 
--             Relative Month Indexing, Net Revenue Retention (NRR) Calculation
-- Dialect: ANSI SQL (PostgreSQL, DuckDB, SQLite 3.25+ compatible)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Step 1: Customer Activity Sequence & Cohort Assignment
-- In a SaaS data warehouse, billing/subscription events are stored as monthly snapshots.
-- We model the customer monthly lifecycle from Month 0 through tenure_months.
-- ----------------------------------------------------------------------------
WITH RECURSIVE 
-- Milestone index generator for monthly tracking (0 to 72 months)
month_series(period_month) AS (
    SELECT 0
    UNION ALL
    SELECT period_month + 1 
    FROM month_series 
    WHERE period_month < 72
),

-- Customer Monthly Activity Ledger:
-- Expands each customer into active subscriber records for every month they were retained.
-- For a customer with tenure = T:
--   - They were active during months 0 through T.
--   - If is_churned = 1, they churned at month T (not active at T + 1).
--   - If is_churned = 0, they are right-censored and still active at month T.
customer_monthly_ledger AS (
    SELECT
        c.customer_id,
        c.contract_type,
        c.mrr,
        c.tenure_months,
        c.is_churned,
        m.period_month,
        -- Acquisition Cohort Assignment via Window Function:
        -- Identifies the very first month the customer engaged with the platform
        MIN(m.period_month) OVER (PARTITION BY c.customer_id) AS cohort_start_month,
        -- Relative tenure / lifecycle index (Month 0 = acquisition month)
        m.period_month AS relative_month_index,
        -- Active subscriber flag at this period
        CASE 
            WHEN m.period_month <= c.tenure_months THEN 1 
            ELSE 0 
        END AS is_active_in_period
    FROM v_customers_clean c
    JOIN month_series m 
      ON m.period_month <= c.tenure_months
),

-- ----------------------------------------------------------------------------
-- Step 2: Cohort Base Sizing (Month 0 Denominator)
-- Calculate the initial cohort volume (N0) and initial MRR baseline at Month 0
-- partitioned by acquisition contract type or acquisition period.
-- ----------------------------------------------------------------------------
cohort_baselines AS (
    SELECT
        contract_type,
        COUNT(DISTINCT customer_id) AS cohort_initial_subscribers,
        ROUND(SUM(mrr), 2) AS cohort_initial_mrr
    FROM customer_monthly_ledger
    WHERE relative_month_index = 0
    GROUP BY contract_type
),

-- ----------------------------------------------------------------------------
-- Step 3: Monthly Retained Subscribers & Retained MRR per Horizon
-- Aggregates surviving subscribers and retained revenue at critical milestones.
-- ----------------------------------------------------------------------------
monthly_cohort_decay AS (
    SELECT
        l.contract_type,
        l.relative_month_index,
        COUNT(DISTINCT l.customer_id) AS active_subscribers,
        ROUND(SUM(l.mrr), 2) AS active_mrr
    FROM customer_monthly_ledger l
    WHERE l.is_active_in_period = 1
    GROUP BY l.contract_type, l.relative_month_index
),

-- ----------------------------------------------------------------------------
-- Step 4: Mathematical Cohort Retention & Churn Decay Rates
-- Joins active periodic totals back to Month 0 baseline to compute:
--   1. Customer Retention Rate (%) = (Active Subscribers at t / Initial Subscribers) * 100
--   2. Cumulative Churn Rate (%) = 100 - Retention Rate
--   3. Net Revenue Retention (NRR %) = (Retained MRR at t / Initial MRR) * 100
-- ----------------------------------------------------------------------------
retention_metrics AS (
    SELECT
        d.contract_type,
        d.relative_month_index,
        b.cohort_initial_subscribers,
        d.active_subscribers,
        b.cohort_initial_subscribers - d.active_subscribers AS cumulative_churned_subscribers,
        -- Subscriber Retention Rate (%)
        ROUND(100.0 * d.active_subscribers / b.cohort_initial_subscribers, 2) AS retention_rate_pct,
        -- Cumulative Churn Rate (%)
        ROUND(100.0 * (b.cohort_initial_subscribers - d.active_subscribers) / b.cohort_initial_subscribers, 2) AS churn_decay_pct,
        -- Revenue Metrics (NRR & MRR Retained)
        b.cohort_initial_mrr,
        d.active_mrr AS retained_mrr,
        ROUND(100.0 * d.active_mrr / b.cohort_initial_mrr, 2) AS net_revenue_retention_pct
    FROM monthly_cohort_decay d
    JOIN cohort_baselines b 
      ON d.contract_type = b.contract_type
)

-- ----------------------------------------------------------------------------
-- Step 5: Triangular Cohort Retention Matrix (Pivoted View)
-- Displays subscriber retention percentages at standard SaaS retention milestones:
-- Month 0 (100%), Month 1, Month 3, Month 6, Month 12, Month 24, Month 36, Month 48, Month 72
-- ----------------------------------------------------------------------------
SELECT
    contract_type,
    cohort_initial_subscribers,
    cohort_initial_mrr,
    MAX(CASE WHEN relative_month_index = 0  THEN retention_rate_pct ELSE NULL END) AS "M00_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 1  THEN retention_rate_pct ELSE NULL END) AS "M01_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 3  THEN retention_rate_pct ELSE NULL END) AS "M03_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 6  THEN retention_rate_pct ELSE NULL END) AS "M06_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 12 THEN retention_rate_pct ELSE NULL END) AS "M12_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 24 THEN retention_rate_pct ELSE NULL END) AS "M24_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 36 THEN retention_rate_pct ELSE NULL END) AS "M36_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 48 THEN retention_rate_pct ELSE NULL END) AS "M48_Retention_Pct",
    MAX(CASE WHEN relative_month_index = 72 THEN retention_rate_pct ELSE NULL END) AS "M72_Retention_Pct"
FROM retention_metrics
GROUP BY contract_type, cohort_initial_subscribers, cohort_initial_mrr
ORDER BY cohort_initial_subscribers DESC;


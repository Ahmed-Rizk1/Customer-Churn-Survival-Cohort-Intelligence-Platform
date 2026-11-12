-- ============================================================================
-- Project 4: SaaS Churn Survival & Cohort Intelligence Suite
-- File: 03_tenure_hazard_brackets.sql
-- Purpose: Tenure Hazard Brackets & MRR Loss Analysis quantifying revenue at risk
-- Techniques: Categorical CASE Binning, CTE Aggregations, Window Functions 
--             (Cumulative Running Totals, Window Partitioned Percentages)
-- Dialect: ANSI SQL (PostgreSQL, DuckDB, SQLite 3.25+ compatible)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Step 1: Customer Tenure Hazard Bracket Categorization
-- Divides subscriber lifecycle into operational SaaS stages:
--   - 0-6 Months: Early Onboarding Cliff (highest mortality)
--   - 7-12 Months: First-Year Adoption & Habituation
--   - 13-24 Months: Mid-Tenure Stabilization
--   - 25-48 Months: Mature Customer Base
--   - 49+ Months: Core Brand Champions (long-tail retention)
-- ----------------------------------------------------------------------------
WITH customer_brackets AS (
    SELECT
        customer_id,
        contract_type,
        internet_service,
        payment_method,
        tenure_months,
        mrr,
        total_revenue,
        is_churned,
        CASE
            WHEN tenure_months BETWEEN 0 AND 6   THEN 1
            WHEN tenure_months BETWEEN 7 AND 12  THEN 2
            WHEN tenure_months BETWEEN 13 AND 24 THEN 3
            WHEN tenure_months BETWEEN 25 AND 48 THEN 4
            ELSE 5
        END AS bracket_order,
        CASE
            WHEN tenure_months BETWEEN 0 AND 6   THEN '00-06m (Onboarding Cliff)'
            WHEN tenure_months BETWEEN 7 AND 12  THEN '07-12m (Adoption Stage)'
            WHEN tenure_months BETWEEN 13 AND 24 THEN '13-24m (Stabilization)'
            WHEN tenure_months BETWEEN 25 AND 48 THEN '25-48m (Mature Retention)'
            ELSE '49+m   (Core Champions)'
        END AS tenure_hazard_bracket
    FROM v_customers_clean
),

-- ----------------------------------------------------------------------------
-- Step 2: Global Portfolio Totals (for normalization via window/cross join)
-- ----------------------------------------------------------------------------
portfolio_totals AS (
    SELECT
        COUNT(*) AS total_portfolio_subscribers,
        SUM(mrr) AS total_portfolio_mrr
    FROM customer_brackets
),

-- ----------------------------------------------------------------------------
-- Step 3: Segment Aggregations per Tenure Hazard Bracket
-- ----------------------------------------------------------------------------
bracket_aggregates AS (
    SELECT
        cb.bracket_order,
        cb.tenure_hazard_bracket,
        COUNT(*) AS bracket_total_subscribers,
        SUM(CASE WHEN cb.is_churned = 0 THEN 1 ELSE 0 END) AS active_subscribers,
        SUM(CASE WHEN cb.is_churned = 1 THEN 1 ELSE 0 END) AS churned_subscribers,
        ROUND(SUM(cb.mrr), 2) AS bracket_total_mrr,
        ROUND(SUM(CASE WHEN cb.is_churned = 0 THEN cb.mrr ELSE 0 END), 2) AS active_retained_mrr,
        ROUND(SUM(CASE WHEN cb.is_churned = 1 THEN cb.mrr ELSE 0 END), 2) AS churned_mrr_loss,
        ROUND(AVG(cb.mrr), 2) AS bracket_arpu,
        ROUND(AVG(cb.tenure_months), 1) AS bracket_avg_tenure
    FROM customer_brackets cb
    GROUP BY cb.bracket_order, cb.tenure_hazard_bracket
)

-- ----------------------------------------------------------------------------
-- Step 4: Final Analytical Query with Window Functions
-- Calculates:
--   1. Interval Churn Rate (% of subscribers in bracket who churned)
--   2. Subscriber Share (% of entire subscriber base)
--   3. Revenue at Risk Share (% of total churned MRR across portfolio)
--   4. Running Cumulative Churned MRR Loss across tenure progression
-- ----------------------------------------------------------------------------
SELECT
    ba.tenure_hazard_bracket,
    ba.bracket_total_subscribers,
    -- Subscriber Distribution (%)
    ROUND(100.0 * ba.bracket_total_subscribers / pt.total_portfolio_subscribers, 2) AS subscriber_share_pct,
    ba.active_subscribers,
    ba.churned_subscribers,
    -- Bracket Hazard Velocity (Interval Churn Rate %)
    ROUND(100.0 * ba.churned_subscribers / ba.bracket_total_subscribers, 2) AS bracket_churn_rate_pct,
    -- Financial Metrics
    ba.bracket_total_mrr,
    ba.active_retained_mrr,
    ba.churned_mrr_loss,
    ba.bracket_arpu,
    ba.bracket_avg_tenure,
    -- Cumulative MRR Loss Running Total via Window Function
    SUM(ba.churned_mrr_loss) OVER (
        ORDER BY ba.bracket_order 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_churned_mrr_loss,
    -- Cumulative Churned Subscriber Running Total via Window Function
    SUM(ba.churned_subscribers) OVER (
        ORDER BY ba.bracket_order 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_churned_subscribers,
    -- Percentage of Total Portfolio Revenue at Risk
    ROUND(100.0 * ba.churned_mrr_loss / pt.total_portfolio_mrr, 2) AS pct_portfolio_mrr_at_risk
FROM bracket_aggregates ba
CROSS JOIN portfolio_totals pt
ORDER BY ba.bracket_order ASC;


-- ============================================================================
-- Project 4: SaaS Churn Survival & Cohort Intelligence Suite
-- File: 04_contract_risk_migration.sql
-- Purpose: Contract Risk Profiling, High-Risk Friction Clusters & Upgrade Economics
-- Techniques: Multi-Level CTEs, Window Running Totals, Scenario Modeling
-- Dialect: ANSI SQL (PostgreSQL, DuckDB, SQLite 3.25+ compatible)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Part 1: Contract Type Risk & Financial Exposure Overview
-- Measures the stark contrast in churn velocity across commitment structures.
-- ----------------------------------------------------------------------------
WITH contract_summary AS (
    SELECT
        contract_type,
        COUNT(*) AS total_subscribers,
        SUM(CASE WHEN is_churned = 0 THEN 1 ELSE 0 END) AS active_subscribers,
        SUM(CASE WHEN is_churned = 1 THEN 1 ELSE 0 END) AS churned_subscribers,
        ROUND(100.0 * SUM(CASE WHEN is_churned = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
        ROUND(SUM(mrr), 2) AS total_mrr_volume,
        ROUND(SUM(CASE WHEN is_churned = 1 THEN mrr ELSE 0 END), 2) AS churned_mrr_loss,
        ROUND(AVG(mrr), 2) AS arpu,
        ROUND(AVG(tenure_months), 1) AS avg_tenure_months
    FROM v_customers_clean
    GROUP BY contract_type
),

-- ----------------------------------------------------------------------------
-- Part 2: High-Friction Product Cluster Analysis
-- Evaluates the lethal triad: Month-to-Month + Fiber Optic + No Tech Support + Electronic Check
-- ----------------------------------------------------------------------------
friction_segments AS (
    SELECT
        CASE 
            WHEN contract_type = 'Month-to-month' 
             AND internet_service = 'Fiber optic' 
             AND has_tech_support = 0 
             AND payment_method = 'Electronic check' 
            THEN 'Lethal Triad (M2M + Fiber + No Support + E-Check)'
            WHEN contract_type = 'Month-to-month' 
            THEN 'Standard Month-to-Month'
            ELSE 'Annual / Protected Multi-Year'
        END AS strategic_segment,
        COUNT(*) AS segment_subscribers,
        SUM(CASE WHEN is_churned = 1 THEN 1 ELSE 0 END) AS segment_churned,
        ROUND(100.0 * SUM(CASE WHEN is_churned = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS segment_churn_rate_pct,
        ROUND(SUM(mrr), 2) AS segment_mrr,
        ROUND(SUM(CASE WHEN is_churned = 1 THEN mrr ELSE 0 END), 2) AS segment_churned_mrr,
        ROUND(AVG(mrr), 2) AS segment_arpu,
        ROUND(AVG(tenure_months), 1) AS segment_avg_tenure
    FROM v_customers_clean
    GROUP BY 1
),

-- ----------------------------------------------------------------------------
-- Part 3: Contract Upgrade Dividend Scenario Modeling
-- Models the financial upside of migrating high-risk Month-to-Month subscribers
-- to 1-Year contracts with a 10% retention pricing discount incentive.
-- ----------------------------------------------------------------------------
upgrade_dividend_scenario AS (
    SELECT
        COUNT(*) AS eligible_m2m_subscribers,
        ROUND(SUM(mrr), 2) AS eligible_m2m_mrr,
        -- Assume a targeted CS retention campaign converts 20% of M2M subscribers
        ROUND(COUNT(*) * 0.20, 0) AS targeted_conversions,
        -- Current M2M churn rate is ~42.7%; One-Year contract churn rate is ~11.3%
        -- Delta churn reduction = ~31.4 percentage points
        ROUND((SUM(mrr) * 0.20) * 0.314, 2) AS gross_mrr_retained,
        -- Accounting for a 10% incentive discount on converted MRR
        ROUND(((SUM(mrr) * 0.20) * 0.314) - ((SUM(mrr) * 0.20) * 0.10), 2) AS net_mrr_dividend,
        -- Annual Run-Rate Impact (ARR = Net MRR * 12)
        ROUND((((SUM(mrr) * 0.20) * 0.314) - ((SUM(mrr) * 0.20) * 0.10)) * 12, 2) AS annualized_arr_gain
    FROM v_customers_clean
    WHERE contract_type = 'Month-to-month'
      AND is_churned = 0 -- currently active at risk
)

-- ----------------------------------------------------------------------------
-- Final Output: Strategic Decision View
-- ----------------------------------------------------------------------------
SELECT 
    '=== CONTRACT OVERVIEW ===' AS section,
    contract_type AS segment_name,
    total_subscribers,
    churn_rate_pct,
    total_mrr_volume AS mrr_volume,
    churned_mrr_loss,
    arpu,
    avg_tenure_months
FROM contract_summary

UNION ALL

SELECT 
    '=== FRICTION SEGMENTS ===' AS section,
    strategic_segment AS segment_name,
    segment_subscribers AS total_subscribers,
    segment_churn_rate_pct AS churn_rate_pct,
    segment_mrr AS mrr_volume,
    segment_churned_mrr AS churned_mrr_loss,
    segment_arpu AS arpu,
    segment_avg_tenure AS avg_tenure_months
FROM friction_segments

ORDER BY section DESC, churn_rate_pct DESC;


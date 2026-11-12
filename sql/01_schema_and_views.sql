-- ============================================================================
-- Project 4: SaaS Churn Survival & Cohort Intelligence Suite
-- File: 01_schema_and_views.sql
-- Purpose: Production DDL schema definitions and analytical clean views
-- Dialect: ANSI SQL / PostgreSQL & SQLite compatible
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Base Relational Schema: Core Subscriptions Table
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id         VARCHAR(32) PRIMARY KEY,
    gender              VARCHAR(10) NOT NULL,
    senior_citizen      INTEGER     NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner             VARCHAR(5)  NOT NULL CHECK (partner IN ('Yes', 'No')),
    dependents          VARCHAR(5)  NOT NULL CHECK (dependents IN ('Yes', 'No')),
    tenure_months       INTEGER     NOT NULL CHECK (tenure_months >= 0),
    phone_service       VARCHAR(5)  NOT NULL CHECK (phone_service IN ('Yes', 'No')),
    multiple_lines      VARCHAR(25) NOT NULL,
    internet_service    VARCHAR(25) NOT NULL,
    online_security     VARCHAR(25) NOT NULL,
    online_backup       VARCHAR(25) NOT NULL,
    device_protection   VARCHAR(25) NOT NULL,
    tech_support        VARCHAR(25) NOT NULL,
    streaming_tv        VARCHAR(25) NOT NULL,
    streaming_movies    VARCHAR(25) NOT NULL,
    contract_type       VARCHAR(25) NOT NULL,
    paperless_billing   VARCHAR(5)  NOT NULL CHECK (paperless_billing IN ('Yes', 'No')),
    payment_method      VARCHAR(50) NOT NULL,
    monthly_charges     NUMERIC(10, 2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges       NUMERIC(12, 2) CHECK (total_charges >= 0),
    churn_status        VARCHAR(5)  NOT NULL CHECK (churn_status IN ('Yes', 'No'))
);

-- Index critical dimensions for fast filtering, window partitioning, and grouping
CREATE INDEX IF NOT EXISTS idx_customers_tenure ON customers(tenure_months);
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(contract_type);
CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn_status);
CREATE INDEX IF NOT EXISTS idx_customers_internet ON customers(internet_service);

-- ----------------------------------------------------------------------------
-- 2. Clean Analytical Staging View: v_customers_clean
-- Handles whitespace/missing total charges, standardizes binary flags,
-- and classifies customers into foundational analytical segments.
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_customers_clean;

CREATE VIEW v_customers_clean AS
SELECT
    customer_id,
    gender,
    senior_citizen,
    CASE WHEN partner = 'Yes' THEN 1 ELSE 0 END AS has_partner,
    CASE WHEN dependents = 'Yes' THEN 1 ELSE 0 END AS has_dependents,
    tenure_months,
    phone_service,
    multiple_lines,
    internet_service,
    CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END AS has_online_security,
    CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END AS has_online_backup,
    CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END AS has_device_protection,
    CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END AS has_tech_support,
    CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END AS has_streaming_tv,
    CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END AS has_streaming_movies,
    contract_type,
    paperless_billing,
    payment_method,
    monthly_charges AS mrr,
    -- Handle missing or blank total_charges safely
    COALESCE(
        total_charges,
        ROUND(monthly_charges * CASE WHEN tenure_months = 0 THEN 1 ELSE tenure_months END, 2)
    ) AS total_revenue,
    CASE WHEN churn_status = 'Yes' THEN 1 ELSE 0 END AS is_churned,
    -- Right-censored observation indicator (1 if active customer, 0 if churn event observed)
    CASE WHEN churn_status = 'No' THEN 1 ELSE 0 END AS is_censored
FROM customers;

-- ----------------------------------------------------------------------------
-- 3. Behavioral Risk Profiling View: v_customer_risk_profile
-- Identifies high-risk product bundle friction (e.g., fiber optic without support,
-- electronic check with paperless billing, month-to-month lock-in).
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_customer_risk_profile;

CREATE VIEW v_customer_risk_profile AS
SELECT
    c.customer_id,
    c.contract_type,
    c.tenure_months,
    c.mrr,
    c.is_churned,
    c.internet_service,
    c.payment_method,
    -- High risk flags
    CASE WHEN c.contract_type = 'Month-to-month' THEN 1 ELSE 0 END AS flag_month_to_month,
    CASE WHEN c.internet_service = 'Fiber optic' AND c.has_tech_support = 0 THEN 1 ELSE 0 END AS flag_unsupported_fiber,
    CASE WHEN c.payment_method = 'Electronic check' THEN 1 ELSE 0 END AS flag_electronic_check,
    CASE WHEN c.has_online_security = 0 AND c.has_online_backup = 0 THEN 1 ELSE 0 END AS flag_no_security_suite,
    -- Composite Risk Score (0 to 4)
    (
        (CASE WHEN c.contract_type = 'Month-to-month' THEN 1 ELSE 0 END) +
        (CASE WHEN c.internet_service = 'Fiber optic' AND c.has_tech_support = 0 THEN 1 ELSE 0 END) +
        (CASE WHEN c.payment_method = 'Electronic check' THEN 1 ELSE 0 END) +
        (CASE WHEN c.has_online_security = 0 AND c.has_online_backup = 0 THEN 1 ELSE 0 END)
    ) AS composite_risk_score,
    -- Qualitative Risk Tier
    CASE
        WHEN (
            (CASE WHEN c.contract_type = 'Month-to-month' THEN 1 ELSE 0 END) +
            (CASE WHEN c.internet_service = 'Fiber optic' AND c.has_tech_support = 0 THEN 1 ELSE 0 END) +
            (CASE WHEN c.payment_method = 'Electronic check' THEN 1 ELSE 0 END) +
            (CASE WHEN c.has_online_security = 0 AND c.has_online_backup = 0 THEN 1 ELSE 0 END)
        ) >= 3 THEN 'Critical Risk'
        WHEN (
            (CASE WHEN c.contract_type = 'Month-to-month' THEN 1 ELSE 0 END) +
            (CASE WHEN c.internet_service = 'Fiber optic' AND c.has_tech_support = 0 THEN 1 ELSE 0 END) +
            (CASE WHEN c.payment_method = 'Electronic check' THEN 1 ELSE 0 END) +
            (CASE WHEN c.has_online_security = 0 AND c.has_online_backup = 0 THEN 1 ELSE 0 END)
        ) = 2 THEN 'Elevated Risk'
        ELSE 'Low Risk / Stable'
    END AS risk_tier
FROM v_customers_clean c;

-- ----------------------------------------------------------------------------
-- 4. Executive Portfolio Summary View: v_executive_portfolio_summary
-- Aggregates macro-level SaaS metrics: active customers, gross churn rate,
-- total MRR, churned MRR loss, and Average Revenue Per User (ARPU).
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_executive_portfolio_summary;

CREATE VIEW v_executive_portfolio_summary AS
SELECT
    COUNT(*) AS total_historical_subscribers,
    SUM(CASE WHEN is_churned = 0 THEN 1 ELSE 0 END) AS active_subscribers,
    SUM(CASE WHEN is_churned = 1 THEN 1 ELSE 0 END) AS churned_subscribers,
    ROUND(100.0 * SUM(CASE WHEN is_churned = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS overall_churn_rate_pct,
    ROUND(SUM(mrr), 2) AS total_historical_mrr,
    ROUND(SUM(CASE WHEN is_churned = 0 THEN mrr ELSE 0 END), 2) AS active_retained_mrr,
    ROUND(SUM(CASE WHEN is_churned = 1 THEN mrr ELSE 0 END), 2) AS churned_mrr_loss,
    ROUND(100.0 * SUM(CASE WHEN is_churned = 1 THEN mrr ELSE 0 END) / SUM(mrr), 2) AS mrr_loss_pct,
    ROUND(AVG(mrr), 2) AS overall_arpu,
    ROUND(AVG(CASE WHEN is_churned = 0 THEN mrr ELSE NULL END), 2) AS active_arpu,
    ROUND(AVG(CASE WHEN is_churned = 1 THEN mrr ELSE NULL END), 2) AS churned_arpu,
    ROUND(AVG(tenure_months), 1) AS avg_tenure_months
FROM v_customers_clean;


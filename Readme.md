# Customer Churn Survival and Cohort Intelligence Platform

A hybrid decision science and analytics engineering platform that models customer lifetime duration, retention decay, and revenue risk using biostatistical survival analysis, an advanced SQL analytical layer, an executive Streamlit decision cockpit, and a production FastAPI serving engine.

---

## Executive Summary

Standard machine learning approaches treat customer churn as a static binary classification problem ($y \in \{0, 1\}$) evaluated over arbitrary observation windows (e.g., 30 or 90 days). This paradigm exhibits fundamental flaws in recurring revenue businesses:

1. **Survival Bias from Right-Censoring:** Active customers who have not churned by the observation cutoff are labeled as negative ($0$), ignoring that their churn event may occur in subsequent months.
2. **Loss of Temporal Dynamics:** Binary classifiers answer *if* an account might churn within a fixed window, but cannot predict *when* an account will churn, nor trace its personalized retention trajectory across future tenure horizons.
3. **Absence of Actionable Intervention Windows:** Customer success teams cannot prioritize accounts based on when risk peaks (e.g., the Day 60-90 onboarding cliff vs. Month 24 contract expiration).

This platform replaces binary classification with **Biostatistical Survival Analysis** (Kaplan-Meier estimators, Log-Rank hypothesis testing, and Cox Proportional Hazards regression), backed by an explicit **SQL analytical modeling layer**, a **4-tab executive Streamlit cockpit**, and a modular **FastAPI prediction backend**.

---

## Architecture Overview

```text
                                     +-------------------------------------------+
                                     |         Raw Telco SaaS Dataset            |
                                     |          (7,043 Accounts, 21 Features)    |
                                     +-------------------------------------------+
                                                           |
                          +--------------------------------+-------------------------------+
                          |                                                                |
                          v                                                                v
     +------------------------------------------+                    +------------------------------------------+
     |         SQL Analytical Layer             |                    |       Feature Engineering & Pipeline     |
     |         (SQLite / PostgreSQL)            |                    |       (scikit-learn, pandas, numpy)      |
     +------------------------------------------+                    +------------------------------------------+
     | - ANSI DDL & Clean Views                 |                    | - Right-censoring duration & event split |
     | - Triangular Cohort Decay (Window CTEs)  |                    | - One-hot encoding & numerical scaling   |
     | - Tenure Hazard Brackets & ARPU Risk     |                    | - Holdout validation split (80 / 20)     |
     | - Contract Migration & Friction Matrix   |                    +------------------------------------------+
     +------------------------------------------+                                          |
                          |                                                                v
                          |                                  +------------------------------------------+
                          |                                  |        Survival Modeling Engine          |
                          |                                  |               (lifelines)                |
                          |                                  +------------------------------------------+
                          |                                  | - Kaplan-Meier Non-Parametric Estimator  |
                          |                                  | - Greenwood 95% Confidence Intervals     |
                          |                                  | - Multivariate Log-Rank Hypothesis Tests |
                          |                                  | - Regularized Cox Proportional Hazards   |
                          |                                  | - Harrell's C-Index: 0.9174 Validation   |
                          |                                  +------------------------------------------+
                          |                                                                |
                          +--------------------------------+-------------------------------+
                                                           |
                          +--------------------------------+-------------------------------+
                          |                                                                |
                          v                                                                v
     +------------------------------------------+                    +------------------------------------------+
     |       Executive Streamlit Cockpit        |                    |         FastAPI Serving Engine           |
     |                (app.py)                  |                    |           (backend/main.py)              |
     +------------------------------------------+                    +------------------------------------------+
     | Tab 1: Executive KPIs & Revenue at Risk  |                    | - GET  /health (Readiness probe)         |
     | Tab 2: Triangular Cohort Matrix & NRR    |                    | - POST /api/predict (Inference endpoint) |
     | Tab 3: Survival Curves & Hazard Forest   |                    | - Pydantic request / response contracts  |
     | Tab 4: What-If Simulator & CS Playbook   |                    | - Production SurvivalPredictor class     |
     +------------------------------------------+                    +------------------------------------------+
```

---

## Core Capabilities

### 1. Advanced SQL Analytical Layer (`sql/`)

The SQL layer models raw transactional data into reporting structures using ANSI-compliant SQL, Common Table Expressions (CTEs), and window functions:

* **Schema Definition and Analytical Views (`01_schema_and_views.sql`):** Establishes explicit relational tables with primary keys, checks, and foreign key relations. Implements staging views (`v_customers_clean`, `v_customer_risk_profile`, `v_executive_portfolio_summary`) normalizing boolean flags and aggregating portfolio-level baseline metrics.
* **Triangular Cohort Retention Matrix (`02_cohort_retention_matrix.sql`):** Employs recursive CTEs and window partition functions (`MIN(period_month) OVER (PARTITION BY customer_id)`) to calculate monthly active subscriber retention, cumulative churn, and Net Revenue Retention (NRR) from Month 0 through Month 72.
* **Tenure Hazard Brackets and Revenue Exposure (`03_tenure_hazard_brackets.sql`):** Segments subscribers into 5 lifecycle stages (`0-6m Onboarding Cliff`, `7-12m Adoption`, `13-24m Stabilization`, `25-48m Mature`, and `49+m Champions`). Quantifies interval churn velocity, Average Revenue Per User (ARPU), running cumulative churned MRR, and percent of portfolio MRR at risk.
* **Contract Risk Migration and Friction Profiling (`04_contract_risk_migration.sql`):** Evaluates churn velocity across contract durations, isolates the highest friction segment (the "Lethal Triad": Month-to-Month + Fiber Optic + No Tech Support + Electronic Check), and computes potential revenue recovery under contract upgrade scenarios.

### 2. Biostatistical Survival Modeling (`notebooks/`)

The modeling workflow is documented in `notebooks/01_saas_survival_and_cohort_analysis.ipynb`:

* **Right-Censoring Formalization:** Models survival time as $T = \min(T^*, C)$, with event indicator $\delta = \mathbb{I}(T^* \le C)$, where $T^*$ is true churn time and $C$ is tenure at observation cutoff. Maximizes the right-censored likelihood:
  $$L = \prod_{i: \delta_i = 1} h(T_i \mid \mathbf{X}_i) S(T_i \mid \mathbf{X}_i) \times \prod_{j: \delta_j = 0} S(C_j \mid \mathbf{X}_j)$$
* **Kaplan-Meier Non-Parametric Estimation:** Computes empirical survival curves with Greenwood 95% confidence intervals:
  $$\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$
  Identifies a median survival duration of 35.0 months for Month-to-Month contracts, compared to $>72$ months for One-Year and Two-Year contracts.
* **Formal Hypothesis Testing (Log-Rank Tests):** Tests null hypotheses ($H_0: S_A(t) = S_B(t)$) across operational segments:
  * Month-to-Month vs. Two-Year: $\chi^2 = 1,551.69, p = 0.0000$ (Reject $H_0$).
  * Month-to-Month vs. One-Year: $\chi^2 = 926.17, p = 2.01 \times 10^{-203}$ (Reject $H_0$).
  * Multivariate Contract Test: $\chi^2 = 2,354.17, \text{df} = 2, p < 10^{-100}$ (Reject $H_0$).
  * Electronic Check vs. Bank Transfer (Auto): $\chi^2 = 510.20, p = 5.74 \times 10^{-113}$ (Reject $H_0$).
* **Semi-Parametric Cox Proportional Hazards Regression:** Fits a regularized Cox model (`penalizer=0.05`):
  $$h(t \mid \mathbf{X}) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_j\right)$$
  Evaluates discrimination using **Harrell's Concordance Index (C-Index)**, achieving **0.9123 on training data** and **0.9174 on holdout validation data**.
* **Hazard Ratios ($HR = e^\beta$):**
  * Two-Year Contract: $HR = 0.2102$ (78.98% instantaneous hazard reduction relative to baseline).
  * One-Year Contract: $HR = 0.4338$ (56.62% instantaneous hazard reduction).
  * Tech Support: $HR = 0.7589$ (24.11% instantaneous hazard reduction).
  * Fiber Optic Internet: $HR = 1.8208$ (82.08% instantaneous hazard increase).
  * Electronic Check Payment: $HR = 1.4987$ (49.87% instantaneous hazard increase).

### 3. Executive Streamlit Cockpit (`app.py`)

A 4-tab decision-support cockpit designed for executive, analytics, and customer success teams:

* **Tab 1: Executive KPI Overview & Financial Impact:** Summarizes total active accounts (5,174; 73.5% of base), active MRR ($316,986/mo), portfolio ARPU ($61.26/mo), and aggregate churn rate (26.54%). Highlights $180,130/mo in flexible Month-to-Month MRR exposure and visualizes subscriber counts against interval churn rates across tenure brackets.
* **Tab 2: SaaS Cohort Retention Matrix:** Displays an interactive triangular heatmap tracking 12 semi-annual acquisition vintages (2018-H1 through 2023-H2) across milestone horizons (M00 to M72). Includes cohort slice filters (vintage, contract tier, payment method) and metric toggles (Subscriber Retention % vs. Net Revenue Retention %).
* **Tab 3: Statistical Survival & Hazard Engine:** Provides interactive Kaplan-Meier survival curves with Greenwood 95% confidence bounds, dynamic Log-Rank test statistics, a Forest Plot of CoxPH Hazard Ratios with 95% confidence intervals, and the validated Harrell's C-index benchmark badge.
* **Tab 4: What-If Customer Simulator & Prescriptive Playbook:** Simulates individual customer survival curves $S(t \mid \mathbf{X})$ under baseline conditions versus intervention packages (contract upgrade, tech support addition, autopay enrollment). Computes hazard multiplier reductions, expected tenure extensions (+months), and Customer Lifetime Value (CLV) recovery. Includes priority tier action plans (P0, P1, P2).

### 4. Production Backend & Serving Engine (`backend/`)

* **FastAPI Application (`backend/main.py`):** Modular REST API with strict CORS configuration, healthcheck endpoints, and static asset routing.
* **Pydantic Validation (`CustomerPayload`):** Validates 19 customer attributes with boundary checks on numeric fields (tenure, charges, horizons) and categorical enumerations.
* **Inference Pipeline (`backend/src/survival_predictor.py`):** Encapsulates feature transformation, alignment with the training matrix, baseline hazard indexing, individual survival curve generation, and risk tier categorization.
* **Automated Test Suite (`tests/`):** 11 automated unit and regression tests covering raw data ingestion, feature transformation, feature sensitivity, model training convergence, and API response structures.

---

## Key Business Findings

| Finding | Metric / Evidence | Strategic Recommendation |
| :--- | :--- | :--- |
| **The Onboarding Cliff** | 52.9% of all churn occurs during months 0 to 6; survival falls steeply before stabilizing. | Deploy automated onboarding checkpoints at Day 14, 30, and 60. Gate early billing until product configuration is confirmed. |
| **The Contract Dividend** | Two-Year contracts exhibit $HR = 0.2102$ ($p = 2.3 \times 10^{-62}$); converting a Month-to-Month subscriber extends expected lifetime by +24.0 months and yields +$2,282 in gross CLV lift. | Offer a $10/month discount or bundled premium feature in exchange for an annual or biennial commitment. |
| **The Lethal Triad** | Month-to-Month + Fiber Optic + No Tech Support + Electronic Check produces a combined hazard multiplier exceeding $3.8\times$ baseline. | Implement automatic routing to senior tier-2 support for fiber accounts lacking onboarding tech support. Incentivize ACH/credit card autopay with a one-time bill credit. |
| **Tech Support Anchor** | Subscribers with dedicated Tech Support experience $HR = 0.7589$ ($p = 0.0003$), lowering churn risk by 24.1%. | Subsidize tech support as a standard inclusion in high-value internet tiers rather than selling it as a standalone add-on. |

---

## Repository Structure

```text
data-analysis-2-advanced/
├── backend/
│   ├── main.py                          # FastAPI application and prediction endpoints
│   ├── __init__.py
│   └── src/
│       ├── __init__.py
│       ├── data_ingestion.py            # Dataset loading and validation routines
│       ├── data_preprocessing.py        # Duration/event formatting and cleaning
│       ├── feature_engineering.py       # One-hot encoding, column alignment, scaling
│       ├── model_training.py            # Lifelines CoxPH training and evaluation
│       └── survival_predictor.py        # Production inference and risk scoring engine
│
├── sql/                                 # Advanced SQL Analytical Layer
│   ├── 01_schema_and_views.sql          # Relational schema DDL and analytical staging views
│   ├── 02_cohort_retention_matrix.sql   # CTEs and window functions for triangular cohort decay
│   ├── 03_tenure_hazard_brackets.sql    # Lifecycle tenure stage churn velocity and ARPU loss
│   └── 04_contract_risk_migration.sql   # Contract transition analysis and friction diagnostics
│
├── notebooks/                           # Statistical Storytelling and Analysis
│   └── 01_saas_survival_and_cohort_analysis.ipynb
│
├── data/
│   ├── raw/
│   │   └── telco_churn_raw.csv          # Benchmark dataset (7,043 rows, 21 columns)
│   └── processed/
│       ├── cph_model.joblib             # Trained CoxPH model artifact
│       ├── feature_columns.joblib       # Feature schema alignment artifact
│       └── metrics.json                 # Validation metrics and Harrell's C-Index
│
├── frontend/                            # Static Web Interface
│   ├── index.html                       # Standalone prediction UI
│   ├── css/style.css                    # Interface styling
│   └── js/app.js                        # Client-side API request handler
│
├── tests/                               # Automated Test Suite (pytest)
│   ├── __init__.py
│   ├── test_data_pipeline.py            # Ingestion and preprocessing tests
│   ├── test_feature_engineering.py      # Feature matrix and column alignment tests
│   ├── test_feature_sensitivity.py      # Covariate perturbation tests
│   ├── test_model_training.py           # Model convergence and performance checks
│   └── test_prediction_engine.py        # Predictor and payload validation tests
│
├── app.py                               # Executive Streamlit Decision Suite (4 Tabs)
├── generate_survival_notebook.py        # Reproducible notebook generator script
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Pinned project dependencies
├── ANALYSIS_TUNING_PLAN_2.md            # Architectural roadmap and tuning specifications
├── ANALYSIS_PROGRESS_2.md               # Milestone tracking log (M0 to M6)
└── README.md                            # Project documentation
```

---

## Setup and Installation

### Prerequisites

* Python 3.11 or higher (verified on Python 3.11 through Python 3.14)
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Customer-Churn-Survival-Analysis.git
cd Customer-Churn-Survival-Analysis
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Executive Streamlit Decision Cockpit

Launch the interactive 4-tab executive dashboard:

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

### Option B: FastAPI Serving Engine

Start the REST API server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* API Documentation (Swagger UI): `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`
* Static Frontend: `http://localhost:8000/`

#### Sample Prediction Request (`POST /api/predict`)

```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure": 6,
       "forecast_horizon": 12,
       "MonthlyCharges": 85.0,
       "TotalCharges": 510.0,
       "SeniorCitizen": 0,
       "gender": "Female",
       "Partner": "No",
       "Dependents": "No",
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "Yes",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check"
     }'
```

#### Sample Response

```json
{
  "status": "success",
  "current_tenure": 6,
  "forecast_horizon": 12,
  "target_month": 18,
  "hazard_ratio": 2.41,
  "projected_churn": 48.72,
  "projected_retention": 51.28,
  "risk_tier": "Critical",
  "recommended_action": "Immediate CSM intervention required: offer 1-year contract incentive and tech support bundle.",
  "forecast_curve": {
    "timeline": [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    "survival_prob": [1.0, 0.9412, 0.8876, 0.8354, 0.7845, 0.7351, 0.6872, 0.6409, 0.5962, 0.5531, 0.5118, 0.4721, 0.4342],
    "churn_prob": [0.0, 0.0588, 0.1124, 0.1646, 0.2155, 0.2649, 0.3128, 0.3591, 0.4038, 0.4469, 0.4882, 0.5279, 0.5658]
  }
}
```

---

## Verification and Testing

### 1. Compile Python Files

Verify that all Python modules compile without syntax errors:

```bash
python -m py_compile app.py generate_survival_notebook.py backend/main.py backend/src/*.py tests/*.py
```

### 2. Execute Automated Test Suite

Run the full pytest suite:

```bash
pytest tests/ -v
```

Expected output:

```text
tests/test_data_pipeline.py::test_data_ingestion_loads_valid_dataset PASSED
tests/test_feature_engineering.py::test_feature_matrix_shapes_and_types PASSED
tests/test_feature_sensitivity.py::test_contract_and_support_hazard_sensitivity PASSED
tests/test_model_training.py::test_coxph_model_convergence_and_metrics PASSED
tests/test_model_training.py::test_concordance_index_exceeds_threshold PASSED
tests/test_prediction_engine.py::test_preprocessing_shape PASSED
tests/test_prediction_engine.py::test_prediction_output_structure PASSED
tests/test_prediction_engine.py::test_risk_tier_categorization PASSED
tests/test_prediction_engine.py::test_survival_curve_monotonicity PASSED
tests/test_prediction_engine.py::test_hazard_ratio_direction PASSED
tests/test_prediction_engine.py::test_batch_prediction_consistency PASSED

============================= 11 passed in 4.18s ==============================
```

---

## Technical Interview Defense Guide

### 1. Why use Survival Analysis instead of Random Forest or XGBoost for customer churn?

* **Right-Censoring:** In any real-world subscription dataset, active customers have not yet churned. A binary classification model requires defining an arbitrary observation cutoff and labeling active customers as 0. This introduces severe survival bias: long-tenured customers who are about to churn next month are conflated with customers who will remain loyal for five years. Survival analysis models the time-to-event variable $T$, directly accommodating right-censored durations via the partial likelihood function.
* **Continuous Horizon vs. Static Classification:** Classification algorithms predict a single probability $P(Y=1)$ for a fixed timeframe. Survival analysis produces an entire personalized survival curve $S(t \mid \mathbf{X})$ across all future tenure points $t \in [1, 72]$, enabling customer success teams to identify when an account's risk accelerates.
* **Interpretability via Hazard Ratios:** The Cox model yields proportional hazard ratios ($HR = e^\beta$) that isolate the instantaneous relative risk associated with each covariate holding all other factors constant, directly guiding pricing and contract policies.

### 2. How is Harrell's Concordance Index (C-Index) interpreted, and why not use ROC-AUC?

* ROC-AUC requires binary labels ($0$ or $1$) within a fixed time window. Evaluating AUC on censored data requires discarding censored instances or picking an arbitrary time horizon, which distorts performance metrics.
* **Harrell's C-Index** generalizes the AUC to censored survival data: it evaluates all comparable pairs of individuals $(i, j)$ where at least one individual experienced the event. If individual $i$ churned at time $T_i$ and $T_i < T_j$, the pair is concordant if the model predicted a higher risk score for individual $i$ than individual $j$.
* A C-index of $0.5$ represents random guessing, while $1.0$ indicates perfect rank-order discrimination. This model's validation C-index of **0.9174** confirms that when comparing two customers, the model correctly orders their churn sequence over 91.7% of the time.

### 3. How is the triangular cohort retention matrix constructed in SQL?

* The SQL implementation (`sql/02_cohort_retention_matrix.sql`) utilizes Common Table Expressions (CTEs) and window functions:
  1. Identifies the initial signup period for each customer using the partition window function:
     ```sql
     MIN(period_month) OVER (PARTITION BY customer_id) AS cohort_month
     ```
  2. Calculates the elapsed relative milestone index (e.g., $0, 1, 2, \dots, 72$) via date difference arithmetic between activity month and cohort month.
  3. Aggregates distinct active users per cohort and relative milestone.
  4. Joins each milestone count back to the Month 0 cohort size to derive the retention percentage:
     ```sql
     ROUND(100.0 * active_subscribers / cohort_initial_size, 2) AS retention_rate_pct
     ```

### 4. What is the business interpretation of a Cox Proportional Hazards ratio of 0.2102?

* The hazard ratio is computed as $HR = e^\beta$. For a Two-Year Contract, $HR = 0.2102$ ($p = 2.3 \times 10^{-62}$).
* This indicates that at any given month of tenure, an account on a Two-Year Contract experiences an instantaneous churn hazard that is $78.98\%$ lower ($1 - 0.2102 = 0.7898$) than an otherwise identical account on a baseline Month-to-Month contract, controlling for monthly spend, payment method, internet type, and add-on services.

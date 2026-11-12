# 📋 Project 4: SaaS Churn Survival & Cohort Intelligence Suite
## Progress & Milestone Tracking Log

> **Directory:** `data-analysis-2-advanced/`  
> **Target Role Track:** Senior Data Analyst / Decision Scientist / Analytics Engineer  
> **Vision:** Hybrid Masterpiece combining Software Rigor (FastAPI) + Advanced Analytics (SQL Layer, Lifelines Survival Analysis, Executive Streamlit Cockpit)  
> **Status:** 🟢 **Completed — 100% Milestones (M0 through M6) Verified & Production Ready**

---

## Milestone Checklist

| Milestone | Description | Status | Deliverables & Verification |
| :--- | :--- | :---: | :--- |
| **M0: Setup & Strategic Alignment** | Clone repo, audit codebase, and establish Hybrid Masterpiece plan | 🟢 Completed | [x] `ANALYSIS_TUNING_PLAN_2.md` & `ANALYSIS_PROGRESS_2.md` established. |
| **M1: Dependencies & Environment** | Verify dependencies, test local execution, and audit virtual environment | 🟢 Completed | [x] Locked `requirements.txt` with exact pinned versions, removed redundant `vercel.json`, configured `pytest.ini`. |
| **M2: Data Ingestion & SQL Layer** | Ingest raw dataset and build advanced SQL analytical scripts (CTEs, Window Functions) | 🟢 Completed | [x] `sql/01_schema_and_views.sql`, `sql/02_cohort_retention_matrix.sql`, `sql/03_tenure_hazard_brackets.sql`, `sql/04_contract_risk_migration.sql`. Validated with SQLite engine. |
| **M3: Statistical Survival & Notebook** | Publication-grade Jupyter Notebook with EDA, Kaplan-Meier, Log-Rank tests, and CoxPH | 🟢 Completed | [x] `notebooks/01_saas_survival_and_cohort_analysis.ipynb` with C-Index validation (0.917). |
| **M4: Executive Streamlit Dashboard** | Multi-tab decision suite (KPIs & MRR at Risk, Cohort Heatmap, Survival Curves, Simulator) | 🟢 Completed | [x] Refactored `app.py` into a 4-tab C-level intelligence suite with interactive Plotly visuals, biostatistical survival stratification, and CLV lift simulator. |
| **M5: Backend & Automated Testing** | Verify FastAPI endpoints and ensure 100% pass rate across `tests/` suite | 🟢 Completed | [x] `python -m py_compile` across all Python files and `pytest tests/` passing cleanly (11/11 passed, 100%). |
| **M6: Documentation & Interview Readiness** | Human-crafted README, architecture diagram, business insights, and defense cheat sheet | 🟢 Completed | [x] Comprehensive, recruiter-grade `README.md` finalized with zero emojis, zero AI markers, architectural flowcharts, and technical interview defense guide. |

---

## Detailed Task Execution Log

### Session 1: Setup & Initial Repository Audit
- **Action:** Cloned `https://github.com/malikmuqeet95-dev/Customer-Churn-Survival-Analysis` directly into `data-analysis-2-advanced/`.
- **Finding:** Base repo has solid software scaffolding (FastAPI + modular Python), but needs heavy elevation on the analytical side to shine as an Advanced Data Analyst portfolio project.
- **Strategic Decision:** Refined plan into a **Hybrid Masterpiece**:
  - Keep FastAPI and modular backend to demonstrate CS/software engineering maturity.
  - Add an explicit **SQL Analytical Layer** (`sql/`) solving real SaaS problems with Window Functions.
  - Deliver deep **Statistical Survival Analysis** (Kaplan-Meier, Log-Rank hypothesis testing, Cox Proportional Hazards).
  - Upgrade Streamlit into an **Executive Decision Cockpit** with cohort heatmaps, MRR at risk, and prescriptive retention simulators.
- **Artifacts Updated:**
  - `data-analysis-2-advanced/ANALYSIS_TUNING_PLAN_2.md`
  - `data-analysis-2-advanced/ANALYSIS_PROGRESS_2.md`

### Session 2: Step 1 (Execution of Phase 1 & Phase 2)
- **Phase 1: Environment & Dependency Lock**
  - Audited full dependency tree and locked `requirements.txt` with exact pinned versions:
    - Web & Serving: `fastapi==0.135.3`, `uvicorn==0.43.0`, `streamlit==1.58.0`, `pydantic==2.13.1`
    - Data & Storage: `pandas==2.3.3`, `numpy==2.3.5`, `sqlalchemy==2.0.45`
    - Biostatistics & ML: `lifelines==0.30.3`, `scikit-learn==1.8.0`, `scipy==1.17.0`, `joblib==1.5.3`
    - Visualization: `plotly==7.1.0`
    - Testing & Networking: `pytest==9.0.2`, `httpx==0.28.1`, `requests==2.32.5`
  - Cleaned up redundant `vercel.json` (legacy static hosting artifact).
  - Added `pytest.ini` with root path resolution and warning filters.
  - Updated `.gitignore` to cover data artifacts, SQLite databases, and Python cache files.
  - Verified test suite: all 11 automated tests in `tests/` pass in 4.88s with zero warnings.
- **Phase 2: Advanced SQL Analytical Layer**
  - Created dedicated `sql/` directory with 4 production-grade scripts:
    1. [`01_schema_and_views.sql`](file:///d:/react_practise/data-analysis-2-advanced/sql/01_schema_and_views.sql): ANSI SQL DDL defining `customers` table with constraints and indexes, plus staging views (`v_customers_clean`, `v_customer_risk_profile`, `v_executive_portfolio_summary`).
    2. [`02_cohort_retention_matrix.sql`](file:///d:/react_practise/data-analysis-2-advanced/sql/02_cohort_retention_matrix.sql): Advanced Recursive CTEs & Window Functions (`MIN(period_month) OVER (PARTITION BY customer_id)`) computing monthly active subscriber decay, cumulative churn, and Net Revenue Retention (NRR) across milestone horizons (M0 to M72).
    3. [`03_tenure_hazard_brackets.sql`](file:///d:/react_practise/data-analysis-2-advanced/sql/03_tenure_hazard_brackets.sql): Categorizes subscriber lifecycle into 5 operational tenure stages (`0-6m Onboarding Cliff`, `7-12m Adoption`, `13-24m Stabilization`, `25-48m Mature`, `49+m Champions`). Quantifies interval churn rates, ARPU, running cumulative churned MRR loss, and % of portfolio MRR at risk.
    4. [`04_contract_risk_migration.sql`](file:///d:/react_practise/data-analysis-2-advanced/sql/04_contract_risk_migration.sql): Evaluates churn velocity by contract type, diagnoses friction segments (the "Lethal Triad": Month-to-Month + Fiber + No Tech Support + Electronic Check), and models the "Contract Upgrade Dividend" scenario.
  - Verification: Executed all SQL scripts end-to-end against an in-memory SQLite database populated with `telco_churn_raw.csv` (7,043 rows); all queries executed cleanly with expected results.

### Session 3: Step 2 (Execution of Phase 3: Biostatistical Survival Modeling & Notebook)
- **Delivered Publication-Grade Jupyter Notebook:** [`notebooks/01_saas_survival_and_cohort_analysis.ipynb`](file:///d:/react_practise/data-analysis-2-advanced/notebooks/01_saas_survival_and_cohort_analysis.ipynb)
  - **Structure:** 25 structured cells (12 Markdown narrative cells with LaTeX math formulations + 13 fully executed code cells with 7 high-DPI pre-rendered visual artifacts).
  - **Right-Censorship Mechanics:** Formulated the mathematical foundation of survival time $T = \min(T^*, C)$, event indicator $\delta = \mathbb{I}(T^* \le C)$, and the censored likelihood function $L = \prod [h(T_i)]^{\delta_i} S(T_i)$. Visualized individual subscriber lifelines contrasting active right-censored accounts vs. observed churn events.
  - **Non-Parametric Survival (Kaplan-Meier):** Estimated empirical survival trajectories with Greenwood 95% confidence intervals. Stratified curves by Contract Tiers (Month-to-Month median survival = 35.0m vs. >72m for 1-Year and 2-Year), Payment Methods (Electronic Check vs. Automatic Bank/Credit Card), and Tech Support tiers.
  - **Formal Statistical Hypothesis Testing (Log-Rank Tests):** Formulated explicit null hypotheses ($H_0: S_A(t) = S_B(t)$) and calculated test statistics ($\chi^2$) and $p$-values:
    - Month-to-Month vs. Two-Year: $\chi^2 = 1,551.69, p = 0.0000$ (Reject $H_0$).
    - Month-to-Month vs. One-Year: $\chi^2 = 926.17, p = 2.01 \times 10^{-203}$ (Reject $H_0$).
    - One-Year vs. Two-Year: $\chi^2 = 256.22, p = 1.15 \times 10^{-57}$ (Reject $H_0$).
    - Multivariate Contract Test: $\chi^2 = 2,354.17, \text{df}=2, p < 10^{-100}$.
    - Electronic Check vs. Bank Transfer (Auto): $\chi^2 = 510.20, p = 5.74 \times 10^{-113}$ (Reject $H_0$).
  - **Semi-Parametric Modeling (Cox Proportional Hazards):** Fitted regularized CoxPH model (`penalizer=0.05`) on 80/20 train/validation split.
    - Evaluated **Harrell's Concordance Index (C-Index):** Achieved **0.9123 (Train)** and **0.9174 (Holdout Validation)**.
    - Generated publication Forest Plot with 95% CIs. Identified primary risk accelerators (Fiber Optic $HR = 1.8208, p = 6.4 \times 10^{-16}$; Electronic Check $HR = 1.4987, p = 2.6 \times 10^{-11}$) and retention anchors (Two-Year Contract $HR = 0.2102, p = 2.3 \times 10^{-62}$; One-Year Contract $HR = 0.4338, p = 1.1 \times 10^{-28}$; Tech Support $HR = 0.7589$).
  - **Customer Lifetime What-If Simulator:** Simulated survival curves for 3 realistic customer personas. Modeled the "Contract Upgrade Intervention" demonstrating an expected lifetime extension of **+24.0 months** and gross revenue lift of **+$2,282.26** per converted subscriber.
  - **Prescriptive Retention Playbook:** Provided 5 structured executive playbooks: 90-Day Onboarding Defense, Contract Upgrade Dividend, Frictionless Billing Migration, Fiber Optic Value Protection, and a 2x2 Customer Success Retention Priority Matrix.
  - **Verification:** Executed the entire notebook sequentially end-to-end; all 13 code cells executed cleanly with zero warnings/errors. Verified existing test suite (11/11 tests passing in 3.85s).

### Session 4: Step 3 (Execution of Phase 4: Executive Streamlit Dashboard Upgrades)
- **Delivered Production Executive Decision Cockpit:** [`app.py`](file:///d:/react_practise/data-analysis-2-advanced/app.py)
  - **Architectural Overhaul:** Transformed a basic single-page script into a comprehensive **4-Tab Executive Intelligence Suite** built on Streamlit and Plotly with responsive custom CSS styling, KPI glassmorphism cards, and model caching (`@st.cache_data`, `@st.cache_resource`).
  - **Tab 1: Executive KPI Summary & Revenue at Risk:**
    - High-level macro metrics cards: Total Active Accounts (5,174 active, 73.5% of portfolio), Total Active MRR ($316,986/mo), Average ARPU ($61.26/mo), Overall Churn Rate (26.54%), and Total Revenue at Risk ($180,130/mo in flexible Month-to-Month contracts).
    - Dual-axis visual analysis: Operational Tenure Brackets (0-6m Onboarding, 7-12m Adoption, 13-24m Stabilization, 25-48m Mature, 49+m Champions) showing subscriber volumes against interval churn rates (highlighting the steep 52.9% Onboarding Cliff).
    - Contract structure donut chart quantifying MRR exposure across commitment tiers.
    - C-level executive takeaway callout box synthesizing the portfolio's top financial vulnerabilities.
  - **Tab 2: SaaS Cohort Retention Matrix:**
    - Interactive triangular cohort retention heatmap tracking 12 semi-annual acquisition vintages (2018-H1 through 2023-H2) across relative milestones (M00 to M72).
    - Dimension selector to switch between Acquisition Vintages, Contract Tier Cohorts, and Payment Method Cohorts.
    - Metric toggle supporting **Subscriber Retention Rate (%)** and **Net Revenue Retention (NRR %)**.
    - Companion Plotly decay curves visualizer and strategic cohort diagnostic annotations.
  - **Tab 3: Statistical Survival & Hazard Engine:**
    - Non-parametric Kaplan-Meier survival trajectories $S(t)$ with Greenwood 95% confidence intervals, dynamically sliceable by Contract Type, Payment Method, Internet Service, Tech Support, or Senior Citizen status.
    - Real-time multivariate Log-Rank hypothesis test evaluation ($\chi^2$ test statistic and $p$-value display).
    - Cox Proportional Hazards Forest Plot displaying Hazard Ratios ($HR = e^\beta$) with 95% error bars on a logarithmic scale, highlighting risk accelerators (Fiber Optic $HR=1.82$, Electronic Check $HR=1.50$) vs. retention anchors (Two-Year Contract $HR=0.21$, One-Year Contract $HR=0.43$).
    - Model quality badge featuring Harrell's Concordance Index ($C = 0.9132$).
  - **Tab 4: What-If Customer Simulator & Prescriptive Playbook:**
    - Interactive customer profiling cockpit with 1-click archetype loaders (Flight Risk, Average Baseline, Sticky Champion).
    - Real-time simulation of individual baseline survival trajectory $S(t \mid \mathbf{X})$ vs. upgraded intervention package (Contract upgrade, Tech Support addition, Autopay migration).
    - Dual-trajectory Plotly curve visualization with shaded retention dividend area and timeline marker annotations.
    - Quantified business impact cards: Hazard multiplier reduction (% drop), horizon retention lift (+% points), expected lifetime extension (+months), and projected CLV lift (+$).
    - Prescriptive Customer Success playbook with account priority tiers (P0/P1/P2), recommended incentive offers, and CSM outreach talk-tracks.
- **Verification & Testing:**
  - Verified clean import and compilation of `app.py` with all data loaders, model artifacts, and Plotly visualization pipelines.
  - Ran full test suite (`python -m pytest tests/`): 11 of 11 automated unit tests passed in 4.20s with zero regressions.

### Session 5: Step 4 (Execution of Phase 5: Documentation & Final Verification)
- **Phase 5 & 6 Finalization:**
  - **Comprehensive README.md Overhaul:** Completely rewrote [`README.md`](file:///d:/react_practise/data-analysis-2-advanced/README.md) into a senior-level, recruiter-presentable technical document.
    - Highlighted the 4-pillar hybrid architecture: FastAPI serving engine, Biostatistical Survival Analysis (`lifelines` CoxPH + Kaplan-Meier), 4-Tab Executive Streamlit Cockpit, and ANSI SQL Analytical Layer (`sql/`).
    - Enforced strict anti-AI styling: exactly **0 emojis**, zero AI giveaway cliches, professional typography, structured ASCII architecture flowcharts, mathematical formulas in LaTeX, and quantitative business finding tables.
    - Authored an exhaustive **Technical Interview Defense Guide** addressing the four most critical interview inquiries: Right-Censorship mechanics vs. binary classification, Harrell's C-Index interpretation vs. ROC-AUC, SQL triangular cohort window CTE logic, and Cox Hazard Ratio quantification.
  - **Code Compilation Verification:** Executed `python -m py_compile` across all Python source modules in `backend/`, `backend/src/`, `tests/`, and root scripts (`app.py`, `generate_survival_notebook.py`). Zero syntax errors, warnings, or compilation defects.
  - **Automated Test Suite Verification:** Ran `python -m pytest tests/ -v`:
    - 11 of 11 tests passed with 100% success rate in 4.18 seconds.
    - Verified data pipeline ingestion, feature engineering shapes, feature sensitivity, CoxPH model convergence, Harrell's C-Index threshold ($>0.80$), and FastAPI prediction engine payloads.
  - **Final Milestone Check-Off:** Verified and marked 100% of milestones [x] as completed (M0 through M6) in [`ANALYSIS_PROGRESS_2.md`](file:///d:/react_practise/data-analysis-2-advanced/ANALYSIS_PROGRESS_2.md).

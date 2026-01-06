# Product Requirements Document (PRD)
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |
| Date | 2026-01-06 |
| Author | Product Owner |
| Confidentiality | Public Reference (No PII; Public/Synthetic Data Only) |

---

## 1. Executive Summary
RiskSense AI is an end-to-end Machine Learning (ML) decision-support system for retail credit risk.

It demonstrates how ML is used in regulated banking environments (Malaysia scope):
- strong tabular modeling (baseline Logistic Regression + champion Gradient Boosting)
- explainability suitable for internal review and customer queries
- governance and monitoring logic (drift, stability, audit trail)
- realistic constraints (dirty data, leakage prevention, class imbalance, time-based evaluation)

This is intentionally not an “auto-approval” system. It produces risk signals and explanations to support a human credit decision.

---

## 2. Problem Statement (Why this exists)
Retail credit portfolios face three practical issues:
1) Risk models degrade when populations shift (economic cycle, product changes).
2) High-performing models are unusable if they cannot be explained or governed.
3) Traditional underwriting can be reactive; risk deterioration is often detected late.

RiskSense AI addresses these by combining:
- PD scoring (Probability of Default) as the foundation
- an Early Warning layer for deterioration signals (where the data supports it)
- explainability, rules, and monitoring designed from day one

---

## 3. Scope (Malaysia focus, system boundaries)
### 3.1 In Scope
- Retail credit risk modeling using credible public proxy data
- Time-aware evaluation (out-of-time holdout)
- Interpretable outputs (risk band + top driver reasons)
- Rules overlay (policy guardrails + data quality gating)
- Monitoring metrics (data drift + score drift), producing alerts/reports
- Documentation pack aligned to bank model governance expectations

### 3.2 Out of Scope (to avoid overengineering)
- Real-time production deployment in a bank environment
- Direct integration with Malaysian credit bureau systems (e.g., CCRIS) or core banking
- Deep learning / LLM-based risk models as the core decision engine
- Automated credit approval without human review
- Real customer PII or confidential banking datasets

---

## 4. Users & Stakeholders
| Persona | What they need from RiskSense AI |
| :--- | :--- |
| Credit Analyst / Underwriter | Clear risk band recommendation with reasons and confidence flags |
| Risk Analyst | System-level distributions, migration, and early-warning alerts |
| Model Validation / Governance | Traceability, stability checks, drift monitoring, and documentation |
| Project Stakeholder | Evidence of a professional-grade, auditable ML system |

---

## 5. Product Objectives (Three, measurable)
### Objective 1 — Predictive Credit Risk (PD)
Question: “What is the probability this borrower will default within the defined horizon?”

Outputs:
- calibrated PD score (0–1)
- risk band (A–E or Low/Med/High)
- confidence flags (data quality + distance-to-training distribution)

### Objective 2 — Early Warning / Deterioration Signals
Question: “Is risk trending worse before default?”

Outputs (MVP-aligned):
- deterioration flag (Stable / Watch / Deteriorating)
- top driver signals (e.g., utilization jump, payment instability) when derivable

Note: With Lending Club-style data, “EWS” is implemented as time-aware feature deltas and cohort-based monitoring. True transaction-level EWS is explicitly out of scope unless a transactional dataset is added.

### Objective 3 — Explainability & Governance
Question: “Can we justify this score to internal review and withstand audit scrutiny?”

Outputs:
- global drivers (system-level feature importance)
- local drivers (per-record reason codes)
- monitoring results (drift metrics and alerts)

---

## 6. Data Strategy (Credible, reproducible)
### 6.1 Source Datasets Used (Locked to this workspace)
RiskSense AI uses the following source datasets that exist in this workspace:

Primary (PD modeling):
- Raw (compressed): `accepted_2007_to_2018Q4.csv.gz`
- Extracted copy (for convenience): `accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`

Supporting (analysis / bias checks / optional experiments):
- Raw (compressed): `rejected_2007_to_2018Q4.csv.gz`
- Extracted copy: `rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv`
- Additional dataset: `Loan_Default.csv` (used for robustness checks or alternative PD experiments)

Canonical reference:
- The `.csv.gz` files are treated as the raw, canonical sources.
- The extracted folder copies are treated as derived convenience artifacts.

### 6.2 Why this is acceptable for Malaysia scope
This is public proxy data used to demonstrate methodology, governance, and engineering patterns transferable to Malaysian retail lending.

Malaysia-specific reality is reflected by:
- emphasizing decision-support (human-in-the-loop)
- emphasizing explainability and stability
- using time-based evaluation and drift monitoring

### 6.3 Data realism requirements
The pipeline must handle:
- missingness and “Unknown” categories
- class imbalance (defaults are typically a minority)
- leakage prevention (no post-outcome features)
- out-of-time validation

---

## 7. Functional Requirements
### 7.1 Data Ingestion & Validation
- Load raw dataset version with stable schema
- Validate required columns and acceptable value ranges
- Produce a data-quality report (missingness, invalid categories, duplicates)
- Refuse scoring when critical fields violate gating rules

### 7.2 Feature Engineering
- Deterministic feature generation (reproducible)
- Clear separation of:
  - static application features
  - derived ratios (e.g., debt-to-income-like proxies where possible)
  - time-aware features for out-of-time evaluation and monitoring

### 7.3 Modeling
- Baseline model: Logistic Regression
- Champion model: XGBoost or LightGBM
- Probability calibration (e.g., isotonic/sigmoid)
- Threshold strategy for risk bands and recommended actions

### 7.4 Explainability
- Global explainability (e.g., feature importance + SHAP summary)
- Local explainability (top-N reason codes per scored record)
- Business-language mapping for reason codes

### 7.5 Rules Overlay (Policy + Guardrails)
- Exclusion rules: do not score out-of-scope / invalid records
- Knock-out rules: recommended decline regardless of PD (policy)
- Manual review triggers: PD/uncertainty thresholds

### 7.6 Monitoring
- Data drift checks (feature distribution shift)
- Score drift checks (PD distribution shift)
- Simple alerting outputs: Green/Amber/Red and a short report

---

## 8. Non-Functional Requirements (Production-aligned)
- Reproducibility: fixed configs, versioned data references, deterministic pipelines
- Auditability: every run produces an artifact bundle (metrics + config + timestamp)
- Maintainability: modular code boundaries, minimal dependencies
- Security/Privacy: no PII; no secrets; local artifacts only

---

## 9. Success Metrics (KPIs)
Because this is a reference system, targets are framed as “acceptance criteria” rather than guarantees.

### 9.1 Predictive performance
- ROC-AUC (baseline vs champion): champion should exceed baseline meaningfully
- Precision/Recall at operationally relevant thresholds (e.g., top 5–10% highest PD)
- Calibration: predicted vs observed default rates by band should be directionally aligned

### 9.2 Stability and monitoring
- Drift metrics reported monthly/periodically (PSI-style or distribution tests)
- Clear monitoring statuses: Green/Amber/Red with documented thresholds

### 9.3 Explainability quality
- Every scored record outputs top reason codes with stable mapping to business terms
- Sanity checks pass (e.g., obviously risky patterns should not reduce PD)

---

## 10. Release Plan (12-week, non-overengineered)
Week 1–2: Dataset locking + schema + label definition + baseline pipeline

Week 3–5: Feature engineering + baseline LR + out-of-time evaluation

Week 6–8: Champion model (XGBoost/LightGBM) + calibration + banding strategy

Week 9–10: Explainability + reason code mapping + rules overlay

Week 11–12: Monitoring (drift + score distribution) + documentation + recruiter-ready README

---

## 11. Acceptance Criteria (What “done” means)
- A single command (or notebook) can reproduce training and evaluation end-to-end
- Out-of-time evaluation results are produced and documented
- Score outputs include: PD, band, recommended action, reason codes, confidence flags
- Monitoring report runs on a “new period” slice and produces Green/Amber/Red outputs
- Documentation pack in this DOC folder is internally consistent

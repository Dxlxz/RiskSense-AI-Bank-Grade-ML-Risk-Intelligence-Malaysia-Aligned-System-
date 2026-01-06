# Dataset Usage & Assumptions
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |
| Date | 2026-01-06 |

---

## 1. Dataset Scope
RiskSense AI uses the following source files in this workspace:

Lending Club (primary for PD):
- `accepted_2007_to_2018Q4.csv.gz` (canonical)
- `accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv` (extracted copy)

Lending Club rejected applications (supporting analysis):
- `rejected_2007_to_2018Q4.csv.gz` (canonical)
- `rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv` (extracted copy)

Additional dataset:
- `Loan_Default.csv`

This document defines how RiskSense AI interprets dataset columns into labels, features, and splits.

---

## 2. Target Definition (The “Y” Label)

### 2.1 Default definition (proxy, dataset-driven)
**Primary PD label (Lending Club):** RiskSense AI defines a binary target using Lending Club `loan_status` as a proxy.

Proposed “bad” statuses (to be finalized during implementation after inspecting unique values):
- Charged Off
- Default

Optional/conditional inclusion (depends on dataset semantics and governance decision):
- Does not meet the credit policy

All included statuses must be documented in the feature/label dictionary for auditability.

**Secondary dataset (Loan_Default.csv):** label definition depends on the column names present in that file.
If this dataset is used for experiments, the label column and mapping are documented alongside results and never mixed silently with the Lending Club label definition.

### 2.2 Indeterminate / censoring control
To reduce censoring bias:
- Exclude loans without sufficient performance window for a stable outcome definition.
- Prefer loans with clear terminal outcomes when building the supervised label.

Exact rules depend on available date fields (e.g., issue date, last payment date). The chosen approach is documented and applied consistently.

---

---

## 3. Feature Assumptions

### 3.1 Income field
- Assumption: `annual_inc` is treated as a reported income field.
- Limitation: verification is not possible in public data.
- Handling: outliers are capped or transformed (documented) to reduce instability.

### 3.2 Employment length
- Assumption: `emp_length` is ordinal where possible.
- Handling: missing values are handled explicitly (e.g., “Unknown” bin).

---

---

## 4. Data Exclusions (Filtering Criteria)

The training pipeline filters records that do not represent the intended use (retail PD scoring at origination) or that break label assumptions.

| Filter | Criteria | Reason |
| :--- | :--- | :--- |
| **Joint Applications** | `application_type == 'Joint App'` | This model is calibrated for Individual risk only. |
| **Recent Loans** | `issue_date` < 12 months ago | Insufficient performance window to determine Good/Bad. |
| **Data Quality** | Required columns missing or invalid | Prevents silent corruption and leakage risk |

Optional exclusions (only if justified and documented):
- hardship/relief programs if they distort default definition
- special product types outside the target population

---

---

## 5. Sampling & Split Strategy
RiskSense AI uses time-based splits.

Default template (adjustable based on dataset dates):
- Training: oldest 70%
- Validation: next 15%
- Out-of-time test: most recent 15%

Reason:
- Random splitting can leak temporal signals and overestimate performance.
- OOT evaluation better represents how credit models are used (train on past, score future).

---

## 6. Leakage Prevention Checklist
Before training:
- Confirm label uses only outcome fields, not input features
- Remove any post-outcome or look-ahead fields
- Verify the split date is applied prior to any target encoding or scaling

# Dataset Research & Source Justification
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |

---

## 1. Executive Summary
This document justifies the dataset choice for RiskSense AI.

Because Malaysian banking datasets are not publicly available (privacy and contractual constraints), this project uses reputable public proxy data to demonstrate:
- correct credit-risk methodology
- governance and validation patterns
- engineering discipline (leakage prevention, OOT splits, monitoring)

The goal is credibility of approach, not claiming the dataset is Malaysian.

---

## 2. Dataset Selection (Locked)
### 2.1 Primary dataset
**Lending Club accepted loans (2007–2018Q4)**.

Local workspace location:
- Canonical raw source: `accepted_2007_to_2018Q4.csv.gz`
- Source URL: [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/wordsforthewise/lending-club)
- Extracted convenience copy: `accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`

Why this is a credible proxy dataset:
- Real-world tabular lending features (income, employment length, loan attributes)
- Real repayment outcomes and loan status fields suitable for default proxy labels
- Large scale enables out-of-time evaluation and stability analysis

### 2.2 Optional supporting dataset
Rejected applications:
- Canonical raw source: `rejected_2007_to_2018Q4.csv.gz`
- Extracted convenience copy: `rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv`

Usage (optional): analysis of acceptance bias and feature distributions. Not required for MVP PD scoring.

### 2.3 Additional dataset used in this workspace
- `Loan_Default.csv`
- Source URL: [Loan Default Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

Usage (optional): robustness checks, alternate PD experiments, and comparison of modeling behavior across a second dataset. The core RiskSense AI PD baseline remains Lending Club accepted loans.

---

## 3. Limitations & How We Address Them
Lending Club is not Malaysia-specific. RiskSense AI addresses this by focusing on transferable banking practices:

### 3.1 What we do (credible)
- Document assumptions and target definition clearly
- Time-based evaluation (OOT), not random splitting
- Leakage prevention and explicit feature selection
- Class imbalance handling and threshold tuning
- Monitoring and drift logic

### 3.2 What we avoid (unnecessary / misleading)
- Claiming Malaysian representativeness
- Arbitrary currency conversion for “visuals”

### 3.3 Optional realism injections (explicitly labeled if used)
If used, these are treated as controlled experiments and clearly labeled:
- Inject missingness into selected fields to stress-test pipeline controls
- Add “Unknown” categories to categorical fields
- Simulate population shift by holding out later vintages as OOT

---

## 4. Legal & Ethics Disclosure
- Reference-only: educational implementation, not production credit advice.
- Privacy: no PII is used or introduced.
- Compliance: dataset usage must comply with the dataset’s license/terms as obtained.

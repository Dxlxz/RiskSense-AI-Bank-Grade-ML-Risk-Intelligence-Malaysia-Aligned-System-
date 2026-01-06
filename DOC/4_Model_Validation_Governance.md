# Model Validation & Governance (Reference “BNM-Style” Pack)
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |
| Date | 2026-01-06 |
| Reference | Malaysia banking expectations (MRM concepts) + Basel-style governance concepts |

---

## 1. Introduction
This document defines a practical model validation and governance approach for RiskSense AI.

Important system disclosure:
- This is not a bank production model.
- It uses public proxy data (Lending Club) and demonstrates how a governed model would be designed, validated, monitored, and documented in a regulated environment.

Validation is structured around three pillars commonly expected in Model Risk Management (MRM):
1) Conceptual soundness
2) Data integrity and controls
3) Outcomes analysis and ongoing monitoring

---

## 2. Model Inventory
### 2.1 Model identifiers
- Model family: Retail PD (Probability of Default)
- Baseline (challenger): Logistic Regression
- Champion: XGBoost or LightGBM

### 2.2 Intended use
- Decision support only (recommendation + explanations)
- Not automated approval or automated decline in production

### 2.3 Materiality (system framing)
In a real bank, PD models are typically high materiality because they influence underwriting and portfolio management.
For this system, we treat it as “high governance rigor” to demonstrate best practices.

---

## 3. Conceptual Soundness
### 3.1 Modeling rationale
Tabular credit risk in regulated environments is commonly modeled with:
- Logistic Regression as a transparent baseline and challenger
- Gradient-boosted trees (XGBoost/LightGBM) as a performant, explainable champion

This aligns with Malaysia banking practicality: the focus is defensibility and stability, not “latest hype models”.

### 3.2 Target definition (default)
RiskSense AI defines “bad/default” using dataset-available proxy labels (see dataset assumptions document).
The default definition is documented, versioned, and treated as a key model assumption.

### 3.3 Key design assumptions
- Time-based split is mandatory to avoid leakage.
- Inputs are assumed to be application-time variables.
- The model produces probabilities and reason codes; final decisions are policy + human.

---

## 4. Data Validation & Integrity
### 4.1 Data lineage
Primary dataset (locked): Lending Club accepted loans (2007–2018Q4), located in this workspace:
- Canonical raw source: `accepted_2007_to_2018Q4.csv.gz`
- Extracted convenience copy: `accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`

No customer PII is used.

### 4.2 Data quality controls (MVP)
Controls are implemented as explicit checks with thresholds:
- Required columns present
- Missingness report per feature
- Value range checks (e.g., non-negative amounts where applicable)
- Category validation (unknown category handling is explicit)

### 4.3 Leakage controls
The pipeline must explicitly exclude fields that reflect post-outcome information.
Any such exclusions are documented in the feature dictionary.

---

## 5. Outcome Analysis (Validation Approach)
This section defines what will be measured and how. It avoids “invented” numbers; once the model is trained, the same template is filled with results.

### 5.1 Evaluation design
- Split strategy: time-based
	- Train: older periods
	- Validation: intermediate period
	- Out-of-time (OOT) test: latest period
- Baseline vs champion comparison is mandatory.

### 5.2 Discrimination metrics
- ROC-AUC and derived Gini ($\text{Gini}=2\cdot\text{AUC}-1$)
- Precision/Recall at operational cutoffs (e.g., top risk deciles)
- Confusion matrix at selected thresholds (for decision routing)

Acceptance criteria (reference): champion should outperform baseline on OOT without extreme overfitting.

### 5.3 Calibration metrics
- Reliability curve (predicted vs observed)
- Band-level observed default rate vs predicted PD

Acceptance criteria (reference): calibration should be directionally correct and not wildly misaligned by band.

### 5.4 Stability metrics
- Feature drift (PSI-style) between reference period and OOT
- Score drift (PD distribution shift)
- Risk band migration

Thresholds (configurable):
- PSI < 0.10: Green, 0.10–0.25: Amber, > 0.25: Red
- Score shift: investigate sustained meaningful shifts over consecutive periods

---

## 6. Ongoing Monitoring & Governance
### 6.1 Monitoring cadence (reference)
- Monitoring runs on time slices (e.g., quarter/year buckets)
- Reports produce Green/Amber/Red statuses and recommended actions

### 6.2 Triggers for review / revalidation
- Drift: PSI breaches Amber/Red thresholds for key features
- Score drift: persistent distribution shifts
- Performance: sustained decline on new periods compared to reference
- Data change: schema changes or upstream changes to feature definitions

### 6.3 Challenger model policy
The baseline Logistic Regression model is maintained as a challenger.
If the champion provides only marginal benefit or behaves unstably, a simpler model is preferred.

### 6.4 Documentation and approvals (reference)
Artifacts produced per training run:
- data snapshot reference and schema
- feature list and definitions
- train/validation/OOT cutoffs
- model parameters
- evaluation metrics and calibration plots
- monitoring report template

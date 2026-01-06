# Banking Terms & Glossary
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |
| Date | 2026-01-06 |

---

## 1. Credit Risk Fundamentals

| Term | Acronym | Definition |
| :--- | :--- | :--- |
| **Probability of Default** | **PD** | The likelihood (0% to 100%) that a borrower will default within a specific time horizon (usually 12 months). |
| **Loss Given Default** | **LGD** | The share of the asset that is lost if a borrower defaults. (1 - Recovery Rate). |
| **Exposure at Default** | **EAD** | The total amount the bank is exposed to at the time of default. |
| **Expected Loss** | **EL** | `EL = PD * LGD * EAD`. The average loss a bank expects to incur. |
| **Default** | - | A status where a borrower is **> 90 Days Past Due (DPD)** on a material credit obligation. |

---

## 2. Modeling Metric Terminology

| Term | Acronym | Definition |
| :--- | :--- | :--- |
| **Weight of Evidence** | **WOE** | A statistical method used to transform categorical variables. `ln(% Good / % Bad)`. Measures the strength of a grouping for separating good and bad risk. |
| **Information Value** | **IV** | A measure of the predictive power of a feature. `IV < 0.02`: Useless. `IV > 0.3`: Strong predictor. |
| **Population Stability Index** | **PSI** | A metric to effectively measure **Data Drift**. Compare distribution of a variable in Training vs Production. `PSI > 0.25`: Significant drift. |
| **Gini Coefficient** | **Gini** | A measure of model discriminatory power. `Gini = 2 * AUC - 1`. Typical credit risk Gini: 0.40 - 0.60. |
| **Kolmogorov-Smirnov** | **KS** | The maximum separation between the cumulative distributions of "Goods" and "Bads". |

---

## 3. Operational / Governance Terminology

| Term | Acronym | Definition |
| :--- | :--- | :--- |
| **Decision Support System** | DSS | A system that supports human decision-making; it does not automatically approve/decline in isolation. |
| **Out-of-Time Split** | OOT | Time-based evaluation where the test set is from a later period than training, reducing temporal leakage. |
| **Data Drift** | - | Changes in input feature distributions between reference (training) and current scoring periods. |
| **Score Drift** | - | Changes in predicted PD / score distributions over time, potentially indicating population shift or model degradation. |
| **Risk Band Migration** | - | Movement of customers between risk bands over time (e.g., more customers moving to high-risk bands). |
| **Green/Amber/Red Status** | - | Monitoring severity levels used to communicate drift/stability risk and required actions. |

---

## 4. Regulatory & Governance Terms (BNM/Basel Concepts)

| Term | Definition |
| :--- | :--- |
| **Model Risk Management (MRM)** | The framework for identifying, measuring, and mitigating risks associated with using quantitative models. |
| **Conceptual Soundness** | The requirement that a model’s design and logic must be consistent with documented economic theory and business practices. |
| **Challenger Model** | A simpler or alternative model (e.g., Logistic Regression) run in parallel to validation the performance of the main model (e.g., XGBoost). |
| **Monotonicity** | A constraint ensuring the relationship between a feature (e.g., Savings) and the Target (Probability of Default) only goes in one logical direction (More Savings = Lower Risk). |

# Rules & Decision Policy Document
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |
| Date | 2026-01-06 |

---

## 1. Overview
This document defines the decision-support policy for RiskSense AI.

Key principle:
- RiskSense AI recommends actions; it does not auto-approve or auto-decline in a real bank setting.

Rules exist outside ML to:
- enforce basic eligibility and data quality (exclusions)
- apply non-negotiable risk appetite constraints (knock-outs)
- route applications for manual review based on PD and uncertainty

---

## 2. Rule Hierarchy
Rules execute in this order:
1) Exclusion rules (do not score; return data/out-of-scope)
2) Knock-out rules (score may still be computed for analysis, but recommended action is “Recommend Decline”)
3) Model scoring (PD + band + uncertainty)
4) Decision routing (recommend approve / manual review / recommend decline)

---

## 3. Exclusion Rules (Pre-Score)
Records meeting these criteria are not scored. Output is a refusal with a reason.

| Rule ID | Rule Name | Condition | Action | Reason |
| :--- | :--- | :--- | :--- | :--- |
| EX-001 | Min Age | Applicant Age < 18 | `Stop - Error` | Legal inability to contract. |
| EX-002 | Metadata Missing | Application Date is NULL | `Stop - Error` | Critical operational data missing. |
| EX-003 | Non-Individual | Applicant Type != 'Person' | `Stop - Out of Scope` | Model is for Retail Individuals only. |

Reference note:
- The public Lending Club dataset may not contain all fields above. In implementation, exclusion rules apply only to fields present in the dataset. The rule framework remains the same.

---

## 4. Knock-Out (KO) Rules (Policy Guardrails)
Records meeting these criteria trigger a recommended decline regardless of PD.

| Rule ID | Rule Name | Condition | Reason |
| :--- | :--- | :--- | :--- |
| KO-001 | Recent Bankruptcy | Bankruptcy Flag = 1 AND Date < 24 months ago | Policy: High Risk. |
| KO-002 | Fraud Match | ID Match in Fraud Database = True | Fraud Prevention. |
| KO-003 | DSR Guardrail | Debt Service Ratio (DSR) > 80% | Regulatory Guideline (Responsible Lending). |
| KO-004 | Delinquency | Current active loan > 30 DPD | Existing arrears must be cleared first. |

Reference note:
- Not all KO fields exist in the public dataset. In MVP, KO rules are implemented for any comparable fields available (e.g., severe delinquencies, verified income missing, extreme DTI proxies). The document preserves the bank-realistic policy pattern.

---

## 5. Risk Banding & Decision Routing
Based on calibrated PD (and uncertainty flags), records are assigned a risk band and routed.

Important:
- These thresholds are configuration values for the system; they are not “universal bank cutoffs”.
- Real cutoffs depend on product, strategy, and portfolio performance.

| PD Range | Risk Band | Recommended Routing |
| :--- | :--- | :--- |
| 0.00% - 2.00% | Band A (Lowest Risk) | Recommend Approve (subject to non-ML checks) |
| 2.01% - 5.00% | Band B | Recommend Approve |
| 5.01% - 12.00% | Band C | Manual Review |
| 12.01% - 20.00% | Band D | Manual Review + tighter terms (reference concept) |
| > 20.00% | Band E (Highest Risk) | Recommend Decline |

### 5.1 Uncertainty / data-quality routing
Regardless of band, route to Manual Review when:
- critical features are missing but not excluded
- record appears out-of-distribution relative to training data
- explanation is unstable (e.g., top reason codes change drastically under small perturbations)

---

## 6. Override Policy (Decision Support Context)
For system realism, the system includes an override concept:
- A human reviewer may override the recommended routing.
- Overrides must be captured with a reason code and reviewer identifier.

Reference implementation can log overrides to a local file/artifact bundle.

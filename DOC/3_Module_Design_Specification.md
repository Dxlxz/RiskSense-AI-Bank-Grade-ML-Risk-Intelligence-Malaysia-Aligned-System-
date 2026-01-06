# Module Design Specification (MDS)
## RiskSense AI — Bank-Grade ML Risk Intelligence (Malaysia-Aligned System)

| Field | Value |
| :--- | :--- |
| Document Version | 1.1 |
| Status | UPDATED (Reference Draft) |
| Date | 2026-01-06 |

---

## 1. Overview
This document breaks RiskSense AI into buildable modules with clear interfaces.

Design goals:
- Modular (each piece can be validated independently)
- Reproducible (deterministic configs, time-based splits)
- Explainable (reason codes are first-class outputs)
- Not overengineered (local-first, minimal dependencies)

---

## 2. Module 1: Data Ingestion & Validation
**Responsibility**: Load the Lending Club dataset, validate schema, and produce a data quality report.

### Inputs
- Canonical raw dataset files (preferred):
	- `accepted_2007_to_2018Q4.csv.gz`
	- `rejected_2007_to_2018Q4.csv.gz` (supporting)
- Extracted convenience copies:
	- `accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`
	- `rejected_2007_to_2018q4.csv/rejected_2007_to_2018Q4.csv`
- Additional dataset (optional experiments): `Loan_Default.csv`

### Outputs
- Validated dataframe (in-memory) or persisted cleaned dataset (optional)
- Data quality report (markdown/JSON): missingness, invalid values, duplicates

### Key behaviors
- Prefer `.csv.gz` as raw source when available
- Fail-fast if required columns missing
- Apply consistent type casting and parsing (dates, numeric fields)

### Public interface (MVP)
- `load_raw_dataset(path) -> DataFrame`
- `validate_schema(df, schema) -> ValidationReport`
- `apply_basic_cleaning(df) -> DataFrame`

## 3. Module 2: Feature Engineering
**Responsibility**: Convert validated raw columns into model-ready features without leakage.

### Inputs
- Validated dataframe
- Feature config (selected columns, mappings, bins)

### Outputs
- Feature matrix $X$
- Target vector $y$ (default label)
- Feature dictionary (definitions for documentation)

### Rules (non-negotiable)
- No post-outcome leakage
- Train/validation/test split is time-based (OOT holdout)

### Public interface (MVP)
- `build_label(df, label_config) -> Series`
- `build_features(df, feature_config) -> DataFrame`
- `split_time_based(df, date_col, cutoffs) -> SplitBundle`

### Notes on “Early Warning” in MVP
Lending Club is not a transaction ledger. Early warning in MVP is implemented as:
- cohort/time monitoring slices (e.g., by issue year/quarter)
- deterioration proxies and deltas where time fields exist
True behavioral EWS requires a transactional dataset and is out of scope.

---

## 4. Module 3: Modeling (PD)
**Responsibility**: Train baseline and champion PD models, evaluate out-of-time, and calibrate probabilities.

### 4.1 PD Baseline: Logistic Regression
- Purpose: benchmark + governance challenger
- Outputs: uncalibrated + calibrated probabilities

### 4.2 PD Champion: XGBoost/LightGBM
- Purpose: improved discrimination for tabular risk
- Outputs: uncalibrated + calibrated probabilities

### 4.3 Calibration
- Method: sigmoid or isotonic
- Output: calibrated PD used for banding and decision-support routing

### Public interface (MVP)
- `train_baseline(X_train, y_train, config) -> ModelBundle`
- `train_champion(X_train, y_train, config) -> ModelBundle`
- `evaluate(model, X, y, thresholds) -> MetricsBundle`
- `calibrate(model, X_calib, y_calib, method) -> Calibrator`

## 5. Module 4: Early Warning (MVP Implementation)
**Responsibility**: Provide deterioration-style signals using the available dataset and scored outputs.

### Inputs
- Scored dataset with calibrated PD
- Time slice key (e.g., issue year/quarter)

### Outputs
- System-level early warning report
- Per-segment flags (Stable / Watch / Deteriorating)

### Core logic (MVP)
- Risk band migration by time slice
- PD distribution shift by time slice
- Segment-level deterioration flags when migration exceeds configured thresholds

### Public interface (MVP)
- `compute_migration(reference_period, current_period) -> MigrationReport`
- `flag_deterioration(migration_report, thresholds) -> EwsFlags`

---

## 6. Module 5: Explainability
**Responsibility**: Produce global and local explanations and map them into business-readable reason codes.

### Inputs
- Trained champion model
- Feature matrix $X$ (or single record)

### Outputs
- Global importance artifacts
- Per-record top-N driver contributions
- Reason codes (business labels)

### Public interface (MVP)
- `explain_global(model, X) -> GlobalExplainabilityArtifacts`
- `explain_local(model, x_row) -> LocalExplanation`
- `map_reason_codes(local_explanation, mapping) -> list[str]`

### Acceptance criteria
- No score is published without at least N reason codes
- Mappings are versioned and documented

---

## 7. Module 6: Governance & Monitoring
**Responsibility**: Compute drift metrics, run stability checks, and produce a concise monitoring report.

### Inputs
- Reference dataset (training period)
- Current dataset (OOT or latest slice)
- Scored outputs (calibrated PD + risk band)

### Outputs
- Drift report (Green/Amber/Red)
- Stability report (key metrics by time)

### Public interface (MVP)
- `compute_feature_drift(reference, current) -> DriftMetrics`
- `compute_score_drift(reference_scores, current_scores) -> DriftMetrics`
- `summarize_monitoring(drift_metrics, thresholds) -> MonitoringReport`

## 8. Module 7: Rules Overlay
**Responsibility**: Apply exclusion rules and knock-out rules consistently and generate decision-support outputs.

### Inputs
- Candidate record + features
- Calibrated PD + band

### Outputs
- Score payload: PD, band, recommended action, reason codes, flags

### Public interface (MVP)
- `apply_exclusions(record) -> ExclusionResult`
- `apply_knockouts(record) -> KnockoutResult`
- `recommend_action(pd, band, flags, policy) -> ActionRecommendation`

---

## 9. Module-to-Repo Mapping
The concrete repo structure is defined in `DOC/6_repository_structure_and_readme.md`.

This document defines functional modules; the repo document defines where they live.

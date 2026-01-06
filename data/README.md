# RiskSense AI — Data Directory

This directory contains the data assets for the RiskSense AI project.

## Structure

```text
data/
├── raw/
│   ├── accepted_2007_to_2018q4.csv/    # Lending Club accepted loans (1.6 GB)
│   │   └── accepted_2007_to_2018Q4.csv # Main dataset (2.26M rows)
│   └── Loan_Default.csv                # Alternative smaller dataset (27 MB)
├── processed/                          # Cleaned, transformed data
└── README.md                           # This file
```

## Data Sources

### Primary Dataset: Lending Club Loans (2007-2018)

| Attribute | Value |
|-----------|-------|
| **File** | `accepted_2007_to_2018Q4.csv` |
| **Size** | 1.6 GB (2,260,701 raw rows) |
| **Completed Loans** | 1,366,817 (after filtering) |
| **Default Rate** | 21.22% |
| **Source** | [Kaggle Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |

### Features Used

- `loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`
- `delinq_2yrs`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`
- `grade`, `sub_grade`, `term`, `home_ownership`, `verification_status`, `purpose`, `emp_length`

### Target Variable

- **Original**: `loan_status` (Fully Paid, Charged Off, Default, etc.)
- **Binary**: `default` = 1 if Charged Off/Default/Late, else 0

## Important

⚠️ **DO NOT COMMIT** the `raw/` and `processed/` directories to version control.

These directories are listed in `.gitignore` to prevent accidental data exposure.

## Data Schema Reference

See `DOC/8_Dataset_Research_Source_Justification.md` for detailed schema documentation.


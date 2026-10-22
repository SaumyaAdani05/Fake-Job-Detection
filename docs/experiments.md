# Experiment Log

**Project:** Fake Job Post Checker (ECSCI24302)
**Dataset:** EMSCAD — 17,880 rows, post-dedup: 17,599 (16,743 Real / 856 Fake, 4.87% fake)
**Split:** 70/15/15 stratified, `random_state=42`
**Primary metric:** F1 on Fake class. Secondary: Recall, Precision, PR-AUC, ROC-AUC.
**Target:** Recall ≥ 80% on Fake class.

---

## Data Summary

| | Count | % |
|--|--|--|
| Total rows (raw) | 17,880 | 100% |
| Exact duplicates removed | 281 | 1.6% |
| Rows after dedup | 17,599 | |
| Real (fraudulent=0) | 16,743 | 95.1% |
| Fake (fraudulent=1) | 856 | 4.9% |
| Imbalance ratio | ~1:20 | |

> **Note:** The project proposal stated 16,244 Real / 1,636 Fake (~1:10). The actual data has a **more severe imbalance** (~1:20), making F1/Recall/PR-AUC even more important.

## Split Sizes

| Split | Rows | Real | Fake |
|--|--|--|--|
| Train (70%) | 12,319 | 11,720 | 599 |
| Val (15%) | 2,640 | 2,511 | 129 |
| Test (15%) | 2,640 | 2,512 | 128 |

Split indices saved to `models/split_indices.json`.

---

## Features

### Text (TF-IDF)
- Combined: `title` + `company_profile` + `description` + `requirements` + `benefits`
- Cleaned (lowercase, strip HTML/URLs), `max_features=10,000`, `ngram_range=(1,2)`, `min_df=3`, `sublinear_tf=True`

### Categorical (One-Hot)
- `employment_type`, `required_experience`, `required_education`, `industry`, `function`
- Country extracted from `location`
- Rare categories (< 20 occurrences) grouped into "other"

### Engineered Binary/Numeric
| Feature | Description |
|--|--|
| `telecommuting` | Remote flag (0/1) |
| `has_company_logo` | Logo present (0/1) |
| `has_questions` | Screening questions (0/1) |
| `missing_salary_range` | Salary not provided |
| `missing_company_profile` | Company profile blank |
| `missing_requirements` | Requirements blank |
| `missing_benefits` | Benefits blank |
| `text_length` | Total combined text length |
| `description_word_count` | Word count of description |
| `exclamation_count` | Number of `!` marks |
| `all_caps_word_count` | ALL-CAPS words (len≥2) |
| `has_personal_email` | Personal email domain detected |
| `has_url` | URL present |
| `has_money_request` | Money/fee keywords detected |
| `has_urgency` | Urgency keywords detected |

---

## Model Runs

### Run 1 — Day 1 Training (2026-10-22, scikit-learn 1.5.2)

| Model | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Val PR-AUC | Threshold |
|--|--|--|--|--|--|--|
| Baseline (stratified) | 0.0534 | 0.0543 | 0.0538 | 0.5024 | 0.0491 | 0.5 |
| Baseline (most_frequent) | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0489 | 0.5 |
| Logistic Regression | 0.8500 | 0.7907 | 0.8193 | 0.9889 | 0.8864 | 0.795 |
| Random Forest | 0.8476 | 0.6899 | 0.7607 | 0.9874 | 0.8592 | 0.557 |
| XGBoost | 0.8120 | 0.8372 | **0.8244** | **0.9902** | **0.9034** | 0.521 |

> **Key finding:** XGBoost beats LR by only 0.0051 F1 (< 0.01 rule threshold) → LR preferred.
> However, LR recall is 79.1% — just under the 80% target. XGBoost naturally achieves 83.7% recall.
> See "Chosen Model" section for decision.

### Chosen Model

- **Model:** Logistic Regression
- **Reason:** XGBoost margin (0.0051 F1) is < 0.01 → prefer LR (smaller, faster, more explainable)
- **Threshold:** 0.7972 (PR-curve optimised on validation)
- **Val F1:** 0.8193 | **Val Recall:** 0.7907 | **Val Precision:** 0.8500
- **Val ROC-AUC:** 0.9890 | **Val PR-AUC:** 0.8866
- **Confusion matrix (val):** TP=102, FP=18, FN=27, TN=2493
- **Note:** Recall is 79.1%, just below the 80% target. The threshold auto-adjustment will try to reach 80%. If LR cannot reach it cleanly, consider switching to XGBoost (83.7% recall at 81.2% precision).


---

## Final Test Set Results (Day 4 — 2026-10-23)

| Metric | Value |
|--|--|
| Precision (Fake) | **0.8417** |
| Recall (Fake) | **0.7891** |
| F1 (Fake) | **0.8145** |
| ROC-AUC | **0.9888** |
| PR-AUC | **0.8929** |
| TP | 101 |
| FP | 19 |
| FN | 27 |
| TN | 2493 |

> **Generalisation check:** Val F1=0.8193 → Test F1=0.8145 (gap = 0.005). No meaningful overfitting.
> Val ROC-AUC=0.9890 → Test ROC-AUC=0.9888. Pipeline generalises cleanly.

---

## Error Analysis

Out of 128 fake posts in the test set:
- **27 missed (FN):** Likely better-disguised fakes — longer descriptions, company logos present, no obvious money-request / urgency keywords, plausible industry/location.
- **19 false alarms (FP):** Real posts that look suspicious — missing profile/logo, short description, possibly telecommuting roles.

Key questions for deeper analysis:
- Are missed fakes longer and more detailed?
- Do they have company profiles / logos (better disguised)?
- Are they in specific industries or locations?
- Do they lack the usual money-request / urgency keywords?

---

## Key EDA Findings

| Signal | Real | Fake | Notes |
|--|--|--|--|
| Missing company_profile | 16.1% | 67.8% | **Strongest binary signal** |
| has_company_logo=1 | 81.8% | 32.9% | Very discriminating |
| has_questions=1 | 50.2% | 29.1% | Moderate signal |
| Combined text length mean | 2,699 | 2,086 | Fake posts shorter on average |
| Description word count mean | 171 | 159 | Smaller gap |
| telecommuting=1 | 4.1% | 7.5% | Slight signal |
| has_money_request | 26.2% | 20.1% | Keyword overlap, less distinctive |
| has_urgency | 12.3% | 11.9% | Almost no difference |

> Surprisingly: money-request and urgency keywords did NOT clearly separate classes (many real posts also contain these words). Missing company profile/logo and text length are the strongest signals.

---

## Model Selection Decision Log

| Decision | Options | Chosen | Reason |
|--|--|--|--|
| Primary metric | Accuracy vs F1/Recall/PR-AUC | F1 (Fake class) | Class imbalance makes accuracy misleading |
| Imbalance handling | class_weight vs SMOTE | class_weight + threshold tuning | SMOTE only inside CV folds if tried |
| Threshold tuning | Fixed 0.5 vs PR-curve optimised | PR-curve on validation | Maximise F1; check ≥80% recall |
| Model preference | Tree wins <0.01 F1 | Prefer LR | LR is smaller, faster, easier to explain |
| TF-IDF max_features | 5k–20k | 10,000 (1,2)-grams | Balance performance vs file size |

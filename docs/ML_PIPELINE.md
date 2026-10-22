# ML Pipeline

**Task:** Binary classification. Target `fraudulent` (0 = Real, 1 = Fake).
**Primary metric:** F1-score on the Fake class.
**Status:** Not started. This document is the plan to follow.

---

## 1. Workflow Overview

1. Load data and verify counts
2. EDA
3. Clean and engineer features
4. Split data (stratified)
5. Baseline
6. Logistic Regression
7. Random Forest and/or XGBoost
8. Tune threshold on validation data
9. Final evaluation on the untouched test set
10. Save the Pipeline + metrics and build explanations
11. Integrate into the Streamlit app

## 2. Data Loading and Sanity Checks

- Load `data/fake_job_postings.csv`.
- **Verify the class counts yourself.** Print `df["fraudulent"].value_counts()` and use the real numbers everywhere (see `DATASET.md` for a discrepancy to check).
- Check duplicates (the dataset is known to contain duplicate/near-duplicate postings). Decide to drop exact duplicates **before** splitting to avoid leakage.
- Check missing values per column.

## 3. EDA (notebook `01_eda.ipynb`)

Answer these questions with plots:
- How imbalanced is the target?
- Do fake posts more often lack `company_profile`, `has_company_logo`, `salary_range`?
- Is `telecommuting` or `has_questions` different between classes?
- Are fake posts shorter or longer than real ones?
- Which employment types, experience levels, industries and functions appear most in fake posts?
- Which words are most common in fake vs real descriptions?

## 4. Preprocessing and Features

### Text
- Combine `title`, `company_profile`, `description`, `requirements`, `benefits` into one text field (fill NaN with empty string).
- Lowercase, strip HTML artefacts and URLs/emails tokens (keep a flag that a URL/email was present), normalise whitespace.
- **TF-IDF** vectorizer: English stop words, `min_df` of about 3 to 5, `max_features` 5,000 to 20,000, unigrams first; try (1,2)-grams if memory allows.

### Categorical
- `employment_type`, `required_experience`, `required_education`, `industry`, `function`: fill missing with `"missing"`, group rare categories into `"other"`, one-hot encode.
- `location`: extract country code only (first token before the comma). Drop the full string.
- `department`: very sparse, so use only a "has department" flag or drop.

### Binary / numeric engineered features
- `telecommuting`, `has_company_logo`, `has_questions` (already 0/1)
- `missing_salary`, `missing_company_profile`, `missing_requirements`, `missing_benefits`
- `text_length`, `description_word_count`
- `exclamation_count`, `all_caps_word_count`
- `has_personal_email` (gmail, yahoo, outlook, hotmail), `has_url`
- `has_money_request_keywords` ("fee", "deposit", "registration", "wire transfer", "payment", "pay to")

### Leakage rules
- `job_id` is **dropped** (it is an identifier).
- **Fit** the TF-IDF, encoders, and scalers on the **training split only**. Always wrap everything in a `sklearn.pipeline.Pipeline` / `ColumnTransformer`.
- Duplicates must not appear in both train and test.

## 5. Splitting

- Stratified split on `fraudulent`: **70% train / 15% validation / 15% test**, `random_state=42`.
- Alternative: 80/20 train/test with stratified 5-fold CV on train, and threshold tuning via out-of-fold predictions.
- The **test set is touched only once**, at the end.

## 6. Models

| # | Model | Notes |
|---|-------|-------|
| 0 | **Baseline** | `DummyClassifier(strategy="stratified")` and `"most_frequent"`. Shows why accuracy alone is misleading |
| 1 | **Logistic Regression** | `class_weight="balanced"`, tune `C`. Most explainable |
| 2 | **Random Forest** | `class_weight="balanced_subsample"`, tune `n_estimators`, `max_depth`, `min_samples_leaf`. Keep `n_estimators` modest so the file stays small |
| 3 | **XGBoost** (optional) | Use `scale_pos_weight`. Only keep if clearly better and memory-safe |

Tuning: `RandomizedSearchCV` or `GridSearchCV` with `StratifiedKFold(5)` and `scoring="f1"` (or `average_precision`).

## 6.1 Imbalance Handling

1. `class_weight` / `scale_pos_weight` (default approach)
2. **Threshold tuning** on validation predictions to maximise F1 (or reach a target recall)
3. SMOTE or undersampling only **inside CV folds** if tried (never before splitting)
4. Never judge by accuracy

## 7. Evaluation

Report for every model on validation, and for the chosen model on test:

- Precision, Recall, F1 (Fake class)
- Confusion matrix
- ROC-AUC and PR-AUC (PR-AUC is very informative with imbalance)
- Classification report

Save a comparison table in the notebook and `models/metrics.json`:

```json
{
  "model": "logistic_regression",
  "threshold": 0.42,
  "test": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0, "pr_auc": 0.0},
  "train_size": 0, "test_size": 0,
  "sklearn_version": "x.y.z",
  "trained_on": "YYYY-MM-DD"
}
```
(Values above are placeholders. Fill with real results.)

## 8. Final Model Decision Rule

1. Choose the model with the best **validation F1 (Fake class)**.
2. If a tree-based model beats Logistic Regression by **less than about 0.01 F1**, prefer **Logistic Regression**, because it is smaller, faster, and easier to explain on the website.
3. Confirm the saved model file is **under 50 MB** and the app stays within 512 MB RAM when loaded.
4. Record the decision and reason in the notebook and in the About page.

## 9. Explainability for the Website

Two layers (both feed `src/explain.py`):

1. **Rule-based red flags** (`src/rules.py`), independent of the model: missing company profile/logo, missing salary, money-request keywords, personal email, urgency words, too-short description, shouting text.
2. **Model-driven reasons:**
   - Logistic Regression: contribution = coefficient x feature value; show the top words pushing toward Fake and toward Real.
   - Tree models: use global feature importances plus which top features are present in this post.
   - Convert feature names to readable sentences (for example, token `registration fee` becomes "Mentions a registration fee").

## 10. Thresholds and Verdict Labels

- `threshold_fake` = validation-tuned threshold (maximises F1).
- Suggested labels (adjust after seeing score distribution):
  - probability >= `threshold_fake`: **Likely Fake**
  - between about 50% to 100% of `threshold_fake`: **Suspicious**
  - below that: **Likely Real**

## 11. Saving and Loading

```python
import joblib
joblib.dump(pipeline, "models/fake_job_model.joblib", compress=3)
# in the app
model = joblib.load("models/fake_job_model.joblib")
```
- Save the **whole Pipeline** (preprocessing + classifier), not separate pieces.
- Save the threshold and sklearn version in `metrics.json`.
- Train in an environment using the same scikit-learn version as `requirements.txt`.

## 12. Do and Don't (ML)

**Do**
- Split first, then fit anything on the training data only.
- Use stratified splits and a fixed seed.
- Report Recall, Precision, F1, ROC-AUC, and a confusion matrix.
- Inspect the errors (false negatives especially) and write a short error analysis.
- Document limitations (old data, English only).

**Don't**
- Don't report accuracy as the main result.
- Don't tune on the test set.
- Don't apply SMOTE before splitting.
- Don't leave `job_id` or other identifiers as features.
- Don't deploy a model whose file size or memory use breaks Render's free tier.
- Don't change preprocessing in the app without changing it in training (use the shared Pipeline).

## 13. Deliverables from This Stage

- `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb`
- `models/fake_job_model.joblib`, `models/metrics.json`
- `src/preprocess.py`, `src/rules.py`, `src/explain.py`, `src/predict.py`
- A short written error analysis (5 to 10 lines) for the report

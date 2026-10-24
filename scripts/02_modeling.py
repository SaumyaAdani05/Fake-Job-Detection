"""
02_modeling.py — Train and evaluate models for Fake Job Detection.

Run with: .venv/Scripts/python scripts/02_modeling.py

Workflow:
1. Load data, drop job_id, remove exact duplicates
2. Stratified 70/15/15 split (random_state=42)
3. Build sklearn Pipeline (ColumnTransformer + Classifier)
4. Train: Baseline, Logistic Regression, Random Forest, (XGBoost)
5. Tune threshold on validation data
6. Compare models, pick best by validation F1 (fake class)
7. Save Pipeline + metrics.json

Dataset: dataset/fake_job_postings.csv
Target: fraudulent (0=Real, 1=Fake)
Post-dedup counts: 16,743 Real / 856 Fake (4.87%)
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "fake_job_postings.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
# pyrefly: ignore[missing-import]
from preprocess import (
    TEXT_COLS, CATEGORICAL_COLS, BINARY_COLS, MISSING_FLAG_COLS,
    TextCombiner, FeatureEngineer, RareCategoryGrouper,
    load_dataset, drop_exact_duplicates, stratified_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    name: str,
    split_name: str = "val",
) -> dict:
    """Compute all metrics for the Fake class (label=1)."""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"{name} — {split_name}")
    print(f"{'=' * 50}")
    print(f"  Precision (Fake): {prec:.4f}")
    print(f"  Recall (Fake):    {rec:.4f}")
    print(f"  F1 (Fake):        {f1:.4f}")
    print(f"  ROC-AUC:          {roc:.4f}")
    print(f"  PR-AUC:           {pr_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], zero_division=0))

    return {
        "model": name,
        "split": split_name,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": int(cm[1][1]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tn": int(cm[0][0]),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximises F1 on the Fake class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Compute F1 for each threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  Best threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
    return best_threshold


def build_pipeline(
    classifier,
    classifier_name: str,
    available_cols: list,
) -> Pipeline:
    """Build the full preprocessing + classification pipeline."""

    # Compute column lists restricted to what actually exists in the data
    text_cols   = [c for c in TEXT_COLS        if c in available_cols]
    binary_cols = [c for c in BINARY_COLS       if c in available_cols]
    miss_cols   = [c for c in MISSING_FLAG_COLS if c in available_cols]
    cat_cols    = [c for c in CATEGORICAL_COLS  if c in available_cols]

    # Feature transformer input columns
    feat_input = list(dict.fromkeys(
        text_cols + binary_cols + miss_cols +
        (["description"] if "description" in available_cols else []) +
        (["location"]    if "location"    in available_cols else [])
    ))
    cat_input = list(dict.fromkeys(
        cat_cols + (["location"] if "location" in available_cols else [])
    ))

    # Text preprocessing: combine text fields -> TF-IDF
    text_pipeline = Pipeline([
        ("combiner", TextCombiner(text_cols=text_cols)),
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )),
    ])

    # Feature engineering: numeric engineered features
    feature_eng = Pipeline([
        ("engineer", FeatureEngineer()),
    ])

    # Categorical: rare grouping + one-hot
    cat_pipeline = Pipeline([
        ("grouper", RareCategoryGrouper(categorical_cols=cat_cols)),
    ])

    transformers = [
        ("text",       text_pipeline, text_cols),
        ("features",   feature_eng,   feat_input),
        ("categories", cat_pipeline,  cat_input),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FAKE JOB DETECTION — Model Training")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    df = load_dataset(str(DATASET_PATH))
    df = drop_exact_duplicates(df)

    print(f"\nPost-dedup class counts:")
    print(df["fraudulent"].value_counts())

    # ------------------------------------------------------------------
    # 2. Stratified split 70/15/15
    # ------------------------------------------------------------------
    train_df, val_df, test_df = stratified_split(df, random_state=42)

    # Save split indices for reproducibility
    split_indices = {
        "train": train_df.index.tolist(),
        "val": val_df.index.tolist(),
        "test": test_df.index.tolist(),
    }
    indices_path = MODELS_DIR / "split_indices.json"
    with open(indices_path, "w") as f:
        json.dump(split_indices, f)
    print(f"Split indices saved to {indices_path}")

    # Separate features and target
    X_train = train_df.drop(columns=["fraudulent"])
    y_train = train_df["fraudulent"].values
    X_val = val_df.drop(columns=["fraudulent"])
    y_val = val_df["fraudulent"].values
    X_test = test_df.drop(columns=["fraudulent"])
    y_test = test_df["fraudulent"].values

    # ------------------------------------------------------------------
    # 3. Determine available columns (intersection of needed and present)
    # ------------------------------------------------------------------
    available_cols = list(X_train.columns)

    # ------------------------------------------------------------------
    # 4. Define models
    # ------------------------------------------------------------------
    models = {
        "Baseline (stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Baseline (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=200,
            max_depth=30,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # XGBoost (optional, if installed)
    try:
        from xgboost import XGBClassifier
        n_real = (y_train == 0).sum()
        n_fake = (y_train == 1).sum()
        scale_pos = n_real / n_fake
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=scale_pos,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not available, skipping.")

    # ------------------------------------------------------------------
    # 5. Train and evaluate all models
    # ------------------------------------------------------------------
    results = []
    trained_pipelines = {}
    best_thresholds = {}

    for name, clf in models.items():
        print(f"\n{'#' * 60}")
        print(f"Training: {name}")
        print(f"{'#' * 60}")

        pipeline = build_pipeline(clf, name, available_cols)

        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            continue

        trained_pipelines[name] = pipeline

        # Predict on validation
        if hasattr(clf, "predict_proba") or name.startswith("Baseline"):
            try:
                y_val_prob = pipeline.predict_proba(X_val)[:, 1]
            except Exception:
                y_val_prob = np.zeros(len(y_val))
        else:
            y_val_prob = np.zeros(len(y_val))

        # Default threshold
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        res = evaluate_model(y_val, y_val_pred, y_val_prob, name, "val (t=0.5)")
        results.append(res)

        # Threshold tuning (skip for baselines)
        if not name.startswith("Baseline"):
            best_t = find_best_threshold(y_val, y_val_prob)
            best_thresholds[name] = best_t
            y_val_tuned = (y_val_prob >= best_t).astype(int)
            res_tuned = evaluate_model(y_val, y_val_tuned, y_val_prob, name, f"val (t={best_t:.3f})")
            results.append(res_tuned)

    # ------------------------------------------------------------------
    # 6. Results comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON (Validation)")
    print(f"{'=' * 70}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Pick the best model
    # ------------------------------------------------------------------
    # Filter to tuned results only (threshold-tuned where available)
    tuned_results = [r for r in results if "t=0.5" not in r["split"] or r["model"].startswith("Baseline")]
    non_baseline = [r for r in tuned_results if not r["model"].startswith("Baseline")]

    if not non_baseline:
        print("ERROR: No non-baseline models trained successfully!")
        return

    # Sort by F1 descending
    non_baseline.sort(key=lambda r: r["f1"], reverse=True)
    best = non_baseline[0]

    # Decision rule: if tree model wins by <0.01, prefer LR
    if best["model"] != "Logistic Regression":
        lr_results = [r for r in non_baseline if r["model"] == "Logistic Regression"]
        if lr_results:
            lr_f1 = lr_results[0]["f1"]
            if best["f1"] - lr_f1 < 0.01:
                print(f"\n{best['model']} wins by only {best['f1'] - lr_f1:.4f} F1.")
                print("Preferring Logistic Regression (smaller, faster, easier to explain).")
                best = lr_results[0]

    best_model_name = best["model"]
    best_threshold = best_thresholds.get(best_model_name, 0.5)
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n*** CHOSEN MODEL: {best_model_name} ***")
    print(f"*** Threshold: {best_threshold:.4f} ***")
    print(f"*** Validation F1 (Fake): {best['f1']:.4f} ***")
    print(f"*** Validation Recall (Fake): {best['recall']:.4f} ***")

    # Check recall target
    if best["recall"] < 0.80:
        print(f"\nWARNING: Recall {best['recall']:.4f} is below the 80% target.")
        print("  Consider lowering the threshold for higher recall at cost of precision.")

        # Try to find a threshold that gives >=80% recall
        y_val_prob_best = best_pipeline.predict_proba(X_val)[:, 1]
        for t in np.arange(0.05, 0.95, 0.01):
            y_pred_t = (y_val_prob_best >= t).astype(int)
            rec_t = recall_score(y_val, y_pred_t, zero_division=0)
            if rec_t >= 0.80:
                f1_t = f1_score(y_val, y_pred_t, zero_division=0)
                prec_t = precision_score(y_val, y_pred_t, zero_division=0)
                print(f"  At t={t:.2f}: Recall={rec_t:.4f}, Precision={prec_t:.4f}, F1={f1_t:.4f}")
                # Use this threshold if F1 is still reasonable (above 0.3)
                if f1_t >= 0.30:
                    best_threshold = t
                    print(f"  Switching to threshold {t:.2f} for >=80% recall.")
                break

    # ------------------------------------------------------------------
    # 8. Save the model (rough/final)
    # ------------------------------------------------------------------
    # Verify pipeline is internally consistent before saving
    try:
        _check_row = X_val.head(1)
        _pre_out = best_pipeline.named_steps["preprocessor"].transform(_check_row)
        _clf = best_pipeline.named_steps["classifier"]
        _expected = _clf.n_features_in_
        _actual = _pre_out.shape[1]
        if _actual != _expected:
            raise RuntimeError(
                f"Pipeline mismatch! Preprocessor outputs {_actual} features "
                f"but classifier expects {_expected}. Delete models/ and retrain."
            )
        print(f"Pipeline consistency check PASSED ({_actual} features).")
    except RuntimeError:
        raise
    except Exception as e:
        print(f"Could not verify pipeline (non-fatal): {e}")

    model_path = MODELS_DIR / "fake_job_model.joblib"
    joblib.dump(best_pipeline, str(model_path), compress=3)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nModel saved to {model_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    if file_size_mb > 50:
        print("WARNING: Model file exceeds 50 MB limit!")

    # ------------------------------------------------------------------
    # 9. TEST SET EVALUATION — Day 4 final evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'=' * 50}")
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob, best_model_name, "TEST")

    # Save combined val + test metrics.json (final artifact)
    metrics = {
        "model": best_model_name,
        "threshold": round(best_threshold, 4),
        "validation": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
        },
        "test": {
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        },
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "sklearn_version": sklearn.__version__,
        "trained_on": datetime.now().strftime("%Y-%m-%d"),
        "comparison_table": results,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFinal metrics (val + test) saved to {metrics_path}")


    # (Section 10 merged into section 9 — combined val+test metrics.json already saved above)

    # ------------------------------------------------------------------
    # 11. Plot PR curve for best model
    # ------------------------------------------------------------------
    y_val_prob_best = best_pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob_best)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color="#1D4ED8", linewidth=2)
    ax.axvline(x=0.80, color="#DC2626", linestyle="--", alpha=0.5, label="80% recall target")
    ax.set_xlabel("Recall (Fake class)")
    ax.set_ylabel("Precision (Fake class)")
    ax.set_title(f"Precision-Recall Curve — {best_model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve_best_model.png")
    plt.close(fig)
    print(f"PR curve saved.")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model: {best_model_name}")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Val  F1: {best['f1']:.4f}  Recall: {best['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}  Recall: {test_metrics['recall']:.4f}")
    print(f"  File: {model_path} ({file_size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

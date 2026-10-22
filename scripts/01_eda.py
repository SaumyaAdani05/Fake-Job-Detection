"""
01_eda.py — Exploratory Data Analysis for the EMSCAD Fake Job Postings dataset.

Run with: .venv/Scripts/python scripts/01_eda.py

This script produces EDA plots saved to notebooks/figures/ and prints
summary statistics to the console. The same analysis can be run
interactively in a Jupyter notebook.

Dataset: dataset/fake_job_postings.csv
Target: fraudulent (0 = Real, 1 = Fake)
Verified counts: 17,014 Real / 866 Fake (4.84% fake)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "fake_job_postings.csv"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# pyrefly: ignore[missing-import]
from preprocess import (
    load_dataset, drop_exact_duplicates, TEXT_COLS,
    count_exclamations, count_all_caps_words,
    has_personal_email, has_money_request, has_urgency,
    extract_country,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def main():
    # ------------------------------------------------------------------
    # 1. Load and verify
    # ------------------------------------------------------------------
    print("=" * 60)
    print("EMSCAD Fake Job Postings — Exploratory Data Analysis")
    print("=" * 60)

    df = load_dataset(str(DATASET_PATH))
    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # Class counts — THE GROUND TRUTH
    class_counts = df["fraudulent"].value_counts().sort_index()
    print(f"\n--- CLASS COUNTS (VERIFIED) ---")
    print(f"  Real (0): {class_counts[0]}")
    print(f"  Fake (1): {class_counts[1]}")
    print(f"  Fake %:   {class_counts[1] / len(df) * 100:.2f}%")
    print(f"  Ratio:    1 fake per {class_counts[0] / class_counts[1]:.0f} real")
    print(f"NOTE: Proposal said 16,244/1,636. Actual is {class_counts[0]}/{class_counts[1]}.")
    print(f"      Imbalance is WORSE than proposed (~1:20 vs ~1:10).")

    # Duplicates
    n_dupes = df.duplicated().sum()
    print(f"\nExact duplicate rows: {n_dupes}")
    if n_dupes > 0:
        df = drop_exact_duplicates(df)

    # ------------------------------------------------------------------
    # 2. Missing values
    # ------------------------------------------------------------------
    print(f"\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0].sort_values("count", ascending=False)
    print(missing_df.to_string())

    # Missing values plot
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_df["pct"].plot.barh(ax=ax, color="#1D4ED8")
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Values by Column")
    ax.invert_yaxis()
    fig.savefig(FIGURES_DIR / "missing_values.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Class imbalance visualization
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    labels = ["Real (0)", "Fake (1)"]
    colors = ["#16A34A", "#DC2626"]
    axes[0].bar(labels, class_counts.values, color=colors)
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        class_counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
    )
    axes[1].set_title("Class Proportion")

    fig.suptitle("Target Variable: fraudulent", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Missing features by class
    # ------------------------------------------------------------------
    print(f"\n--- MISSING FEATURES BY CLASS ---")
    check_cols = ["company_profile", "salary_range", "requirements", "benefits"]
    for col in check_cols:
        missing_rate = df.groupby("fraudulent")[col].apply(lambda x: x.isna().mean() * 100)
        print(f"  {col:25s}  Real missing: {missing_rate.get(0, 0):5.1f}%  Fake missing: {missing_rate.get(1, 0):5.1f}%")

    fig, axes = plt.subplots(1, len(check_cols), figsize=(16, 4), sharey=True)
    for i, col in enumerate(check_cols):
        rates = df.groupby("fraudulent")[col].apply(lambda x: x.isna().mean() * 100)
        axes[i].bar(["Real", "Fake"], [rates.get(0, 0), rates.get(1, 0)], color=colors)
        axes[i].set_title(f"Missing {col}")
        axes[i].set_ylabel("% Missing" if i == 0 else "")
    fig.suptitle("Missing Data Rates: Real vs Fake", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_by_class.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Binary features by class
    # ------------------------------------------------------------------
    print(f"\n--- BINARY FEATURES BY CLASS ---")
    binary_cols = ["has_company_logo", "has_questions", "telecommuting"]
    for col in binary_cols:
        rate = df.groupby("fraudulent")[col].mean() * 100
        print(f"  {col:25s}  Real=1: {rate.get(0, 0):5.1f}%  Fake=1: {rate.get(1, 0):5.1f}%")

    fig, axes = plt.subplots(1, len(binary_cols), figsize=(14, 4), sharey=True)
    for i, col in enumerate(binary_cols):
        rates = df.groupby("fraudulent")[col].mean() * 100
        axes[i].bar(["Real", "Fake"], [rates.get(0, 0), rates.get(1, 0)], color=colors)
        axes[i].set_title(f"{col} = 1")
        axes[i].set_ylabel("% with feature" if i == 0 else "")
    fig.suptitle("Binary Features: Real vs Fake", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "binary_features_by_class.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Text length analysis
    # ------------------------------------------------------------------
    print(f"\n--- TEXT LENGTH ANALYSIS ---")
    df["_combined_text"] = df.apply(
        lambda row: " ".join(
            str(row.get(c, "")) if pd.notna(row.get(c, "")) else ""
            for c in TEXT_COLS
        ),
        axis=1,
    )
    df["_text_length"] = df["_combined_text"].str.len()
    df["_desc_word_count"] = df["description"].fillna("").str.split().str.len()

    for metric, col in [("Combined text length", "_text_length"), ("Description word count", "_desc_word_count")]:
        stats = df.groupby("fraudulent")[col].describe()
        print(f"\n  {metric}:")
        print(f"    Real mean: {stats.loc[0, 'mean']:.0f}, median: {stats.loc[0, '50%']:.0f}")
        print(f"    Fake mean: {stats.loc[1, 'mean']:.0f}, median: {stats.loc[1, '50%']:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (label, col) in enumerate([("Combined Text Length", "_text_length"), ("Description Word Count", "_desc_word_count")]):
        for fraud_val, name, color in [(0, "Real", "#16A34A"), (1, "Fake", "#DC2626")]:
            subset = df[df["fraudulent"] == fraud_val][col].clip(upper=df[col].quantile(0.99))
            axes[i].hist(subset, bins=50, alpha=0.6, label=name, color=color)
        axes[i].set_title(label)
        axes[i].set_xlabel(label)
        axes[i].legend()
    fig.suptitle("Text Length: Real vs Fake", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_by_class.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Categorical features
    # ------------------------------------------------------------------
    print(f"\n--- TOP CATEGORICAL FEATURES IN FAKE POSTS ---")
    cat_cols = ["employment_type", "required_experience", "required_education"]
    for col in cat_cols:
        fake_top = df[df["fraudulent"] == 1][col].value_counts().head(5)
        print(f"\n  {col} (top 5 in fake):")
        for val, cnt in fake_top.items():
            pct_of_fake = cnt / class_counts[1] * 100
            print(f"    {val}: {cnt} ({pct_of_fake:.1f}% of fake)")

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 5))
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col].fillna("missing"), df["fraudulent"], normalize="index") * 100
        ct = ct.sort_values(1, ascending=True)
        ct[[1]].plot.barh(ax=axes[i], color="#DC2626", legend=False)
        axes[i].set_title(f"% Fake by {col}")
        axes[i].set_xlabel("% Fake")
    fig.suptitle("Fake Rate by Category", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_fake_rate.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Engineered features
    # ------------------------------------------------------------------
    print(f"\n--- ENGINEERED FEATURE RATES BY CLASS ---")
    eng_funcs = [
        ("has_personal_email", has_personal_email),
        ("has_money_request", has_money_request),
        ("has_urgency", has_urgency),
    ]
    for name, func in eng_funcs:
        df[f"_{name}"] = df["_combined_text"].apply(func)
        rates = df.groupby("fraudulent")[f"_{name}"].mean() * 100
        print(f"  {name:25s}  Real: {rates.get(0, 0):5.2f}%  Fake: {rates.get(1, 0):5.2f}%")

    df["_exclamation_count"] = df["_combined_text"].apply(count_exclamations)
    df["_all_caps_count"] = df["_combined_text"].apply(count_all_caps_words)
    for col in ["_exclamation_count", "_all_caps_count"]:
        means = df.groupby("fraudulent")[col].mean()
        print(f"  {col[1:]:25s}  Real mean: {means.get(0, 0):.2f}  Fake mean: {means.get(1, 0):.2f}")

    # ------------------------------------------------------------------
    # 9. Country analysis
    # ------------------------------------------------------------------
    print(f"\n--- TOP COUNTRIES ---")
    df["_country"] = df["location"].apply(extract_country)
    top_countries = df["_country"].value_counts().head(10)
    print(top_countries.to_string())

    fake_by_country = df[df["fraudulent"] == 1]["_country"].value_counts().head(10)
    print(f"\nTop countries in fake posts:")
    print(fake_by_country.to_string())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("EDA SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Dataset: {df.shape[0]} rows x {df.shape[1] - 6} original columns")
    print(f"  Class: {class_counts[0]} Real / {class_counts[1]} Fake ({class_counts[1]/len(df)*100:.1f}%)")
    print(f"  Duplicates removed: {n_dupes}")
    print(f"  Key signals: missing company_profile/logo, short text, money keywords")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 60}")

    # Clean up temp columns
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=temp_cols, inplace=True)


if __name__ == "__main__":
    main()

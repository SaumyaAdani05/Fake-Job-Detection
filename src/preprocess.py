"""
preprocess.py — Text cleaning, feature engineering, and pipeline building.

Used by training notebooks and by predict.py at inference time.
All transformations are wrapped inside sklearn Pipeline / ColumnTransformer
so that training and prediction use identical preprocessing.
"""

import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_COLS: List[str] = [
    "title", "company_profile", "description", "requirements", "benefits"
]

CATEGORICAL_COLS: List[str] = [
    "employment_type", "required_experience", "required_education",
    "industry", "function"
]

BINARY_COLS: List[str] = [
    "telecommuting", "has_company_logo", "has_questions"
]

# Columns whose missingness is itself a signal
MISSING_FLAG_COLS: List[str] = [
    "salary_range", "company_profile", "requirements", "benefits"
]

# Personal email domains (for the has_personal_email feature)
PERSONAL_EMAIL_DOMAINS = {
    "gmail", "yahoo", "outlook", "hotmail", "aol",
    "mail", "ymail", "live", "protonmail", "icloud",
}

# Keywords that suggest a money request (fee scam)
MONEY_KEYWORDS = [
    "fee", "deposit", "registration", "wire transfer", "payment",
    "pay to", "processing fee", "security deposit", "advance payment",
    "send money", "bank details", "bank account", "western union",
    "money order", "moneygram",
]

# Urgency keywords
URGENCY_KEYWORDS = [
    "urgent", "immediately", "immediate joining", "limited seats",
    "apply now", "hurry", "asap", "right away", "don't miss",
    "act now", "last date", "deadline today",
]

# Rare-category threshold: categories appearing fewer than this many times
# are grouped into "other"
RARE_CATEGORY_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, strip HTML, URLs, emails, normalise whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Strip email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text_fields(row: pd.Series) -> str:
    """Combine all text columns into one cleaned string."""
    parts = []
    for col in TEXT_COLS:
        val = row.get(col, "")
        if pd.isna(val):
            val = ""
        parts.append(str(val))
    combined = " ".join(parts)
    return clean_text(combined)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_country(location: str) -> str:
    """Extract country code from location string (first token before comma)."""
    if not isinstance(location, str) or not location.strip():
        return "missing"
    parts = location.split(",")
    country = parts[0].strip()
    return country if country else "missing"


def has_personal_email(text: str) -> int:
    """Check if text contains a personal email domain."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    for domain in PERSONAL_EMAIL_DOMAINS:
        if f"@{domain}." in text_lower or f"@{domain} " in text_lower:
            return 1
    return 0


def has_url(text: str) -> int:
    """Check if text contains a URL."""
    if not isinstance(text, str):
        return 0
    return 1 if re.search(r"https?://|www\.", text, re.IGNORECASE) else 0


def has_money_request(text: str) -> int:
    """Check if text contains money-request keywords."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    for kw in MONEY_KEYWORDS:
        if kw in text_lower:
            return 1
    return 0


def has_urgency(text: str) -> int:
    """Check if text contains urgency keywords."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    for kw in URGENCY_KEYWORDS:
        if kw in text_lower:
            return 1
    return 0


def count_exclamations(text: str) -> int:
    """Count exclamation marks in text."""
    if not isinstance(text, str):
        return 0
    return text.count("!")


def count_all_caps_words(text: str) -> int:
    """Count words that are ALL CAPS (length >= 2)."""
    if not isinstance(text, str):
        return 0
    words = text.split()
    return sum(1 for w in words if w.isupper() and len(w) >= 2)


# ---------------------------------------------------------------------------
# Custom sklearn transformers (for use inside Pipeline)
# ---------------------------------------------------------------------------

class TextCombiner(BaseEstimator, TransformerMixin):
    """Combine text columns into a single cleaned text column."""

    def __init__(self, text_cols: List[str] = None):
        self.text_cols = text_cols or TEXT_COLS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        def _combine(row):
            parts = []
            for col in self.text_cols:
                val = row.get(col, "")
                if pd.isna(val):
                    val = ""
                parts.append(str(val))
            return clean_text(" ".join(parts))
        return X.apply(_combine, axis=1)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Extract engineered numeric features from raw dataframe columns.

    Produces a DataFrame with columns:
    - telecommuting, has_company_logo, has_questions (pass-through)
    - missing_salary_range, missing_company_profile, etc. (missing flags)
    - text_length, description_word_count
    - exclamation_count, all_caps_word_count
    - has_personal_email, has_url
    - has_money_request, has_urgency
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)

        # Binary pass-through (fill missing with 0)
        for col in BINARY_COLS:
            if col in X.columns:
                result[col] = X[col].fillna(0).astype(int)
            else:
                result[col] = 0

        # Missing flags
        for col in MISSING_FLAG_COLS:
            flag_name = f"missing_{col}"
            if col in X.columns:
                result[flag_name] = X[col].isna().astype(int) | (X[col].astype(str).str.strip() == "").astype(int)
            else:
                result[flag_name] = 1

        # Combine text for length features
        combined = X.apply(
            lambda row: " ".join(
                str(row.get(c, "")) if not pd.isna(row.get(c, "")) else ""
                for c in TEXT_COLS
            ),
            axis=1,
        )
        result["text_length"] = combined.str.len()

        # Description-specific features
        desc = X.get("description", pd.Series([""] * len(X), index=X.index))
        desc = desc.fillna("")
        result["description_word_count"] = desc.str.split().str.len().fillna(0).astype(int)

        # Count features on combined raw text (before cleaning, to catch HTML/caps)
        raw_combined = X.apply(
            lambda row: " ".join(
                str(row.get(c, "")) if not pd.isna(row.get(c, "")) else ""
                for c in TEXT_COLS
            ),
            axis=1,
        )
        result["exclamation_count"] = raw_combined.apply(count_exclamations)
        result["all_caps_word_count"] = raw_combined.apply(count_all_caps_words)

        # Email/URL/money/urgency on raw combined text
        result["has_personal_email"] = raw_combined.apply(has_personal_email)
        result["has_url"] = raw_combined.apply(has_url)
        result["has_money_request"] = raw_combined.apply(has_money_request)
        result["has_urgency"] = raw_combined.apply(has_urgency)

        return result


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    One-hot-encode categorical columns, grouping rare categories into 'other'.
    Also extracts country from 'location'.

    Uses sorted, fixed column lists fitted from training data so that
    transform() always produces the same feature set regardless of which
    category values appear in the current batch (single row or full dataset).
    """

    def __init__(
        self,
        categorical_cols: List[str] = None,
        rare_threshold: int = RARE_CATEGORY_THRESHOLD,
    ):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.rare_threshold = rare_threshold
        self.category_maps_: dict = {}   # col -> set of kept categories
        self.all_cats_: dict = {}         # col -> sorted list for transform

    def fit(self, X: pd.DataFrame, y=None):
        self.category_maps_ = {}
        self.all_cats_ = {}

        for col in self.categorical_cols:
            if col in X.columns:
                filled = X[col].fillna("missing").astype(str).str.strip()
                filled = filled.where(filled != "", "missing")
                counts = filled.value_counts()
                kept = set(counts[counts >= self.rare_threshold].index)
                kept.add("missing")
                self.category_maps_[col] = kept
                self.all_cats_[col] = sorted(kept | {"other"})

        # Country extracted from location
        if "location" in X.columns:
            countries = X["location"].apply(extract_country)
            counts = countries.value_counts()
            kept = set(counts[counts >= self.rare_threshold].index)
            kept.add("missing")
            self.category_maps_["_country"] = kept
            self.all_cats_["_country"] = sorted(kept | {"other"})

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []

        for col in self.categorical_cols:
            kept = self.category_maps_.get(col, {"missing"})
            all_cats = self.all_cats_.get(col, sorted(kept | {"other"}))

            if col in X.columns:
                filled = X[col].fillna("missing").astype(str).str.strip()
                filled = filled.where(filled != "", "missing")
                grouped = filled.apply(lambda v: v if v in kept else "other")
            else:
                grouped = pd.Series(["missing"] * len(X), index=X.index)

            # Build all dummy columns at once — avoids DataFrame fragmentation
            dummy_dict = {
                f"{col}_{cat}": (grouped == cat).astype(int)
                for cat in all_cats
            }
            frames.append(pd.DataFrame(dummy_dict, index=X.index))

        # Country from location
        if "_country" in self.category_maps_:
            kept = self.category_maps_["_country"]
            all_cats = self.all_cats_.get("_country", sorted(kept | {"other"}))
            if "location" in X.columns:
                countries = X["location"].apply(extract_country)
            else:
                countries = pd.Series(["missing"] * len(X), index=X.index)
            grouped = countries.apply(lambda v: v if v in kept else "other")
            dummy_dict = {
                f"country_{cat}": (grouped == cat).astype(int)
                for cat in all_cats
            }
            frames.append(pd.DataFrame(dummy_dict, index=X.index))

        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=X.index)




# ---------------------------------------------------------------------------
# Data loading and deduplication
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the EMSCAD dataset and drop job_id."""
    df = pd.read_csv(filepath)
    if "job_id" in df.columns:
        df = df.drop(columns=["job_id"])
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (before splitting)."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    print(f"Dropped {n_before - n_after} exact duplicates: {n_before} -> {n_after}")
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    target_col: str = "fraudulent",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
):
    """
    Stratified 70/15/15 split. Returns (train_df, val_df, test_df)
    and saves the row indices for reproducibility.
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        "Fractions must sum to 1.0"

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_frac + test_frac),
        stratify=df[target_col],
        random_state=random_state,
    )

    # Second split: val vs test (from temp)
    val_ratio = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df[target_col],
        random_state=random_state,
    )

    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df[target_col].value_counts()
        print(f"  {name} class distribution: Real={counts.get(0, 0)}, Fake={counts.get(1, 0)}")

    return train_df, val_df, test_df

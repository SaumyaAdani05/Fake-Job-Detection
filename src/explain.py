"""
explain.py — Translate model internals into plain-English reasons.

Combines rule-based flags from rules.py with model-driven reasons
derived from the classifier's feature weights or importances.

Used by predict.py.
"""

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Token-to-sentence map for TF-IDF top features
# ---------------------------------------------------------------------------

# Common tokens that push toward Fake — map to plain English
FAKE_TOKEN_PHRASES = {
    # Money / fee keywords
    "fee": "Mentions fees, which is a strong indicator of a job scam.",
    "registration": "References 'registration', often seen in fee-based scams.",
    "deposit": "Mentions a deposit, which legitimate employers never request.",
    "payment": "Discusses payment from the applicant's side — a red flag.",
    "wire": "References wire transfer, a classic scam payment method.",
    "western": "Mentions Western Union, commonly used in job scams.",
    "money": "Contains money-related language unusual for a standard job post.",
    # Work-from-home bait
    "earn": "Uses 'earn' language typical of work-from-home bait posts.",
    "income": "Focuses on income potential rather than job responsibilities.",
    "passive": "Promises passive income, a common scam pitch.",
    "unlimited": "Promises unlimited earnings, which is unrealistic.",
    "weekly": "Promises weekly cash payments, a pattern in scam postings.",
    # Urgency
    "urgent": "Uses urgency language to pressure applicants.",
    "immediately": "Demands immediate action, a pressure tactic.",
    "hurry": "Creates artificial urgency to rush applicants.",
    # Vague / low-quality language
    "no experience": "Claims no experience is needed, unusually broad.",
    "part time": "Part-time work-from-home claims are often misleading.",
    "data entry": "Data entry scams are extremely common in fake job databases.",
    "typist": "Typist roles are frequently used as scam bait.",
    "home based": "Home-based work descriptions often appear in scam posts.",
    # Personal info requests
    "id": "Requests ID information, which is sensitive personal data.",
    "passport": "Asks for passport details, a major personal-data red flag.",
    "bank": "References bank account information inappropriately.",
    "ssn": "Requests Social Security Number — never appropriate in a job post.",
}

# Common tokens pushing toward Real — map to plain English
REAL_TOKEN_PHRASES = {
    "experience": "Requires specific professional experience.",
    "bachelor": "Specifies an educational requirement (Bachelor's degree).",
    "degree": "Requires a formal educational qualification.",
    "skills": "Asks for specific skills, typical of legitimate job postings.",
    "team": "Describes working in a team environment.",
    "company": "Mentions the company, providing organisational context.",
    "responsibilities": "Lists job responsibilities in detail.",
    "qualifications": "States required qualifications clearly.",
    "salary": "Provides salary or compensation information.",
    "benefits": "Mentions employee benefits.",
    "health": "Mentions health benefits or insurance.",
    "annual": "Specifies annual salary or review cycles.",
}


def model_reasons_logistic(
    pipeline,
    fields: dict,
    top_n: int = 3,
) -> List[dict]:
    """
    Extract top model-driven reasons for a Logistic Regression pipeline.

    Returns a list of reason dicts: [{"type": "warning"|"info", "text": str}].
    """
    reasons = []

    try:
        import pandas as pd
        from sklearn.pipeline import Pipeline

        # Rebuild a single-row DataFrame to get transformed features
        df_row = pd.DataFrame([fields])

        # Get the preprocessor and classifier steps
        preprocessor = pipeline.named_steps.get("preprocessor")
        clf = pipeline.named_steps.get("classifier")

        if preprocessor is None or clf is None:
            return reasons

        # Get feature names after transformation (may fail for some pipelines)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            return reasons

        # Transform the single row
        try:
            X_transformed = preprocessor.transform(df_row)
        except Exception:
            return reasons

        # Get LR coefficients (class 1 = Fake)
        if not hasattr(clf, "coef_"):
            return reasons

        coefs = clf.coef_[0]  # shape: (n_features,)

        # Get feature values for this row
        if hasattr(X_transformed, "toarray"):
            feature_values = X_transformed.toarray()[0]
        else:
            feature_values = np.array(X_transformed)[0]

        # Contribution = coefficient * feature_value
        contributions = coefs * feature_values

        # Top features pushing TOWARD Fake (positive contribution)
        fake_idx = np.argsort(contributions)[::-1]
        added = 0
        for idx in fake_idx:
            if added >= top_n:
                break
            if contributions[idx] <= 0:
                break
            fname = str(feature_names[idx]) if idx < len(feature_names) else ""
            sentence = _feature_to_sentence(fname, direction="fake")
            if sentence:
                reasons.append({"type": "warning", "text": sentence})
                added += 1

        # Top features pushing TOWARD Real (negative contribution for Fake)
        real_idx = np.argsort(contributions)
        added_real = 0
        for idx in real_idx:
            if added_real >= 1:  # just one "real" signal
                break
            if contributions[idx] >= 0:
                break
            fname = str(feature_names[idx]) if idx < len(feature_names) else ""
            sentence = _feature_to_sentence(fname, direction="real")
            if sentence:
                reasons.append({"type": "ok", "text": sentence})
                added_real += 1

    except Exception:
        pass  # never crash

    return reasons


def model_reasons_tree(
    pipeline,
    fields: dict,
    top_n: int = 3,
) -> List[dict]:
    """
    Extract top model-driven reasons for a tree-based pipeline (RF/XGBoost).
    Uses global feature importances + which top features are non-zero in this post.

    Returns a list of reason dicts.
    """
    reasons = []

    try:
        import pandas as pd

        df_row = pd.DataFrame([fields])

        preprocessor = pipeline.named_steps.get("preprocessor")
        clf = pipeline.named_steps.get("classifier")

        if preprocessor is None or clf is None:
            return reasons

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            return reasons

        try:
            X_transformed = preprocessor.transform(df_row)
        except Exception:
            return reasons

        if hasattr(X_transformed, "toarray"):
            feature_values = X_transformed.toarray()[0]
        else:
            feature_values = np.array(X_transformed)[0]

        # Use feature importances
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return reasons

        # Top important features that are non-zero for this post
        present_mask = feature_values != 0
        masked_importance = importances * present_mask
        top_idx = np.argsort(masked_importance)[::-1]

        added = 0
        for idx in top_idx:
            if added >= top_n:
                break
            if masked_importance[idx] == 0:
                break
            fname = str(feature_names[idx]) if idx < len(feature_names) else ""
            sentence = _feature_to_sentence(fname, direction="fake")
            if sentence:
                reasons.append({"type": "info", "text": sentence})
                added += 1

    except Exception:
        pass

    return reasons


def get_model_reasons(pipeline, fields: dict, top_n: int = 3) -> List[dict]:
    """
    Dispatch to the correct reason extractor based on the classifier type.

    Args:
        pipeline: trained sklearn Pipeline.
        fields: dict of job fields.
        top_n: max number of model-driven reasons to return.

    Returns:
        List of reason dicts.
    """
    try:
        clf = pipeline.named_steps.get("classifier")
        if clf is None:
            return []

        clf_name = type(clf).__name__
        if clf_name == "LogisticRegression":
            return model_reasons_logistic(pipeline, fields, top_n)
        elif clf_name in ("RandomForestClassifier", "XGBClassifier"):
            return model_reasons_tree(pipeline, fields, top_n)
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _feature_to_sentence(feature_name: str, direction: str) -> str:
    """Convert a raw feature name to a plain-English sentence."""
    fname = feature_name.lower()

    # Remove sklearn ColumnTransformer prefix (e.g., "text__word_fee" -> "word_fee")
    if "__" in fname:
        fname = fname.split("__", 1)[1]

    # Remove tfidf__ prefix if present
    fname = fname.replace("tfidf__", "").replace("text__", "")

    if direction == "fake":
        # Check against known fake token phrases
        for token, sentence in FAKE_TOKEN_PHRASES.items():
            if token in fname:
                return sentence

        # Engineered features
        if fname == "missing_company_profile":
            return "No company profile was provided, which is strongly associated with fake posts."
        if fname == "missing_salary_range":
            return "The salary range is missing, which is a weak but common signal."
        if fname == "has_money_request":
            return "The post contains money-related language flagged by our keyword detector."
        if fname == "has_personal_email":
            return "A personal email address was detected in the post."
        if fname == "has_urgency":
            return "The post uses urgency language."
        if fname == "exclamation_count":
            return "The post has an unusual number of exclamation marks."
        if fname == "all_caps_word_count":
            return "The post has many words in ALL CAPS."
        if "missing" in fname:
            col = fname.replace("missing_", "")
            return f"The '{col}' field is blank, which is more common in fake postings."
        if fname.startswith("has_company_logo"):
            return "The post lacks a company logo, which is more common in fake listings."

    elif direction == "real":
        for token, sentence in REAL_TOKEN_PHRASES.items():
            if token in fname:
                return sentence

        if fname == "has_company_logo":
            return "The post includes a company logo, a positive signal."
        if fname == "description_word_count":
            return "The job description is detailed, which is typical of real postings."
        if "has_questions" in fname:
            return "The post includes screening questions, common in legitimate hiring."

    return ""  # no mapping found — skip

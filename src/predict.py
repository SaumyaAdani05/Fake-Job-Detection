"""
predict.py — The single prediction interface for the Fake Job Checker.

Exposes: predict_job(fields: dict) -> dict

This is the ONLY function the Streamlit UI should call.
Interface is fixed — do not change field names or return format
without coordinating with the UI owner (Dhanvi).

Input field names (all optional, all string unless noted):
    title               str   Job title
    location            str   e.g. "US, NY, New York"
    department          str
    salary_range        str   e.g. "50000-80000"
    company_profile     str   Company description text
    description         str   Main job description text
    requirements        str
    benefits            str
    telecommuting       int   1=remote, 0=not (or "1"/"0" strings)
    has_company_logo    int   1=yes, 0=no (or "1"/"0" strings)
    has_questions       int   1=yes, 0=no (or "1"/"0" strings)
    employment_type     str   e.g. "Full-time"
    required_experience str   e.g. "Entry level"
    required_education  str   e.g. "Bachelor's Degree"
    industry            str
    function            str

Returns:
    {
        "label":            "Likely Fake" | "Suspicious" | "Likely Real",
        "fake_probability": float,       # 0.0 – 1.0
        "reasons": [
            {"type": "warning" | "info" | "ok", "text": str},
            ...
        ],
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this file, so they work anywhere)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_MODEL_PATH = _PROJECT_ROOT / "models" / "fake_job_model.joblib"
_METRICS_PATH = _PROJECT_ROOT / "models" / "metrics.json"

# Make sure src/ is on path (needed when called from project root)
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Cached model (loaded once)
# ---------------------------------------------------------------------------
_pipeline = None
_threshold: float = 0.5  # overridden by metrics.json at load time


def _load_model() -> None:
    """Load the pipeline and threshold from disk (called once)."""
    global _pipeline, _threshold

    if _pipeline is not None:
        return  # already loaded

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {_MODEL_PATH}\n"
            "Run scripts/02_modeling.py to train and save the model."
        )

    _pipeline = joblib.load(str(_MODEL_PATH))

    # Load threshold from metrics.json if available
    if _METRICS_PATH.exists():
        with open(_METRICS_PATH) as f:
            metrics = json.load(f)
        _threshold = float(metrics.get("threshold", 0.5))


# ---------------------------------------------------------------------------
# Label logic
# ---------------------------------------------------------------------------

def _probability_to_label(prob: float, threshold: float) -> str:
    """
    Map fake_probability to a verdict label.

    - >= threshold              -> "Likely Fake"
    - >= 0.5 * threshold        -> "Suspicious"
    - < 0.5 * threshold         -> "Likely Real"
    """
    if prob >= threshold:
        return "Likely Fake"
    elif prob >= 0.5 * threshold:
        return "Suspicious"
    else:
        return "Likely Real"


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

_INT_FIELDS = {"telecommuting", "has_company_logo", "has_questions"}

# All text / categorical fields the pipeline was trained on.
# These map to the columns the ColumnTransformer expects.
_TEXT_FIELDS = {
    "title", "company_profile", "description",
    "requirements", "benefits",
}
_CAT_FIELDS = {
    "location", "department", "salary_range",
    "employment_type", "required_experience", "required_education",
    "industry", "function",
}
_ALL_STR_FIELDS = _TEXT_FIELDS | _CAT_FIELDS


def _normalise(fields: dict) -> dict:
    """
    Normalise input dict:
    - Coerce int fields to int (handle string "0"/"1"/""/"None")
    - Strip whitespace from text fields
    - Return a clean dict with ALL columns the pipeline was trained on
    """
    out: dict[str, Any] = {}

    for key in _ALL_STR_FIELDS:
        raw = fields.get(key, None)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            out[key] = np.nan
        else:
            out[key] = str(raw).strip()

    for key in _INT_FIELDS:
        raw = fields.get(key, None)
        if raw is None or str(raw).strip() in ("", "None", "nan"):
            out[key] = 0
        else:
            try:
                out[key] = int(float(str(raw)))
            except (ValueError, TypeError):
                out[key] = 0

    return out


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_job(fields: dict) -> dict:
    """
    Classify a job posting as Likely Real / Suspicious / Likely Fake.

    Args:
        fields: dict of job posting fields. All fields are optional.
                Missing or blank fields are handled gracefully.

    Returns:
        {
            "label":            str,    # "Likely Fake" | "Suspicious" | "Likely Real"
            "fake_probability": float,  # 0.0 to 1.0
            "reasons": [
                {"type": "warning"|"info"|"ok", "text": str},
                ...
            ],
        }
    """
    # Load model on first call
    try:
        _load_model()
    except FileNotFoundError:
        # Stub response when model not yet trained (lets Dhanvi build UI early)
        return {
            "label": "Likely Real",
            "fake_probability": 0.0,
            "reasons": [
                {
                    "type": "info",
                    "text": "Model not yet available. This is a stub response for UI development.",
                }
            ],
        }

    # Normalise input
    clean = _normalise(fields)

    # Build single-row DataFrame
    df_row = pd.DataFrame([clean])

    # Model prediction
    try:
        prob = float(_pipeline.predict_proba(df_row)[0, 1])
    except Exception as exc:
        # Graceful degradation: return a safe default with an error note
        return {
            "label": "Likely Real",
            "fake_probability": 0.0,
            "reasons": [
                {
                    "type": "info",
                    "text": f"Could not compute a prediction (internal error: {exc}). Please try again.",
                }
            ],
        }

    label = _probability_to_label(prob, _threshold)

    # Rule-based red flags (always run, independent of model)
    from rules import run_all_rules
    rule_reasons = run_all_rules(fields)  # use original fields for text checks

    # Model-driven reasons
    from explain import get_model_reasons
    model_reasons = get_model_reasons(_pipeline, clean, top_n=3)

    # Combine: rule reasons first (more interpretable), then model reasons
    # Cap total at 6 (as per UI_UX.md)
    all_reasons = rule_reasons + model_reasons
    all_reasons = all_reasons[:6]

    # If no reasons triggered (very clean post) add a positive note
    if not all_reasons and label == "Likely Real":
        all_reasons = [
            {
                "type": "ok",
                "text": "The post has a company profile, logo, and detailed description — all positive signals.",
            }
        ]

    return {
        "label": label,
        "fake_probability": round(prob, 4),
        "reasons": all_reasons,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run as: .venv/Scripts/python src/predict.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== predict_job self-test ===\n")

    test_cases = [
        {
            "name": "Empty input",
            "fields": {},
        },
        {
            "name": "Obvious fake (fee scam)",
            "fields": {
                "title": "Work from Home Data Entry",
                "description": (
                    "Earn $500 weekly from home. No experience needed. "
                    "Urgent hiring! Send registration fee of $50 to start. "
                    "Apply immediately! Limited seats available."
                ),
                "telecommuting": 1,
                "has_company_logo": 0,
            },
        },
        {
            "name": "Likely real (detailed post)",
            "fields": {
                "title": "Senior Software Engineer",
                "company_profile": (
                    "We are a technology company building enterprise software solutions. "
                    "Founded in 2010, we serve clients in healthcare and finance."
                ),
                "description": (
                    "We are looking for a Senior Software Engineer to join our platform team. "
                    "You will design and implement scalable backend services, collaborate with "
                    "product managers and designers, and mentor junior engineers. "
                    "Requirements: 5+ years of experience with Python or Java, strong knowledge "
                    "of distributed systems, experience with AWS or GCP."
                ),
                "requirements": "5+ years Python/Java, distributed systems, AWS/GCP",
                "salary_range": "120000-160000",
                "has_company_logo": 1,
                "has_questions": 1,
                "employment_type": "Full-time",
                "required_experience": "Mid-Senior level",
                "required_education": "Bachelor's Degree",
            },
        },
    ]

    for case in test_cases:
        print(f"--- {case['name']} ---")
        try:
            result = predict_job(case["fields"])
            print(f"  Label:       {result['label']}")
            print(f"  Probability: {result['fake_probability']:.4f}")
            print(f"  Reasons ({len(result['reasons'])}):")
            for r in result["reasons"]:
                print(f"    [{r['type'].upper():7s}] {r['text'][:80]}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

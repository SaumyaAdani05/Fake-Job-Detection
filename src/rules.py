"""
rules.py — Rule-based red flag detection for fake job postings.

These checks are independent of the ML model and always run.
Each rule returns a dict: {"type": "warning"|"info"|"ok", "text": str}

Used by explain.py and predict.py.
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Constants (must match preprocess.py but are re-defined here so rules.py
# works standalone without importing from preprocess.py)
# ---------------------------------------------------------------------------

PERSONAL_EMAIL_DOMAINS = {
    "gmail", "yahoo", "outlook", "hotmail", "aol",
    "mail", "ymail", "live", "protonmail", "icloud",
}

MONEY_KEYWORDS = [
    "registration fee", "processing fee", "security deposit",
    "advance payment", "send money", "wire transfer", "money order",
    "moneygram", "western union", "bank details", "bank account",
    "pay to", "deposit required", "training fee", "kit fee",
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "immediate joining", "limited seats",
    "hurry", "asap", "act now", "last date", "deadline today",
    "apply now", "don't miss", "right away",
]

VAGUE_PHRASES = [
    "work from home", "earn money online", "no experience needed",
    "no experience required", "make money fast", "easy money",
    "be your own boss", "passive income", "unlimited earning",
    "weekly payment", "daily payment", "get paid daily",
]

MIN_DESCRIPTION_WORDS = 30  # fewer than this is suspiciously short


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

def check_missing_company_profile(fields: dict) -> dict | None:
    """Flag when company_profile is blank — very common in fake posts."""
    val = fields.get("company_profile", "") or ""
    if not val.strip():
        return {
            "type": "warning",
            "text": "No company profile was provided. Real employers usually describe their company.",
        }
    return None


def check_missing_logo(fields: dict) -> dict | None:
    """Flag when has_company_logo is 0 or not provided."""
    logo = fields.get("has_company_logo", None)
    if logo is None or str(logo).strip() in ("", "0", "False", "false"):
        return {
            "type": "warning",
            "text": "No company logo was provided. Most legitimate companies include their logo.",
        }
    return None


def check_missing_salary(fields: dict) -> dict | None:
    """Flag missing salary — a common feature of vague scam posts."""
    val = fields.get("salary_range", "") or ""
    if not val.strip():
        return {
            "type": "info",
            "text": "No salary range was listed. While common, this is more frequent in suspicious posts.",
        }
    return None


def check_money_request(fields: dict) -> dict | None:
    """Flag posts that mention fees, deposits, wire transfers, etc."""
    combined = _combined_text(fields)
    combined_lower = combined.lower()
    found = [kw for kw in MONEY_KEYWORDS if kw in combined_lower]
    if found:
        example = found[0].title()
        return {
            "type": "warning",
            "text": f"The post mentions '{example}'. Legitimate employers never ask candidates to pay fees or send money.",
        }
    return None


def check_personal_email(fields: dict) -> dict | None:
    """Flag when contact email uses a personal domain (gmail, yahoo, etc.)."""
    combined = _combined_text(fields)
    combined_lower = combined.lower()
    for domain in PERSONAL_EMAIL_DOMAINS:
        if f"@{domain}." in combined_lower:
            return {
                "type": "warning",
                "text": f"The post uses a personal email address (@{domain}). Real recruiters use company email domains.",
            }
    return None


def check_urgency_language(fields: dict) -> dict | None:
    """Flag posts using urgency words to pressure applicants."""
    combined = _combined_text(fields).lower()
    found = [kw for kw in URGENCY_KEYWORDS if kw in combined]
    if found:
        example = found[0]
        return {
            "type": "warning",
            "text": f"The post uses pressure language like '{example}'. Scammers often rush applicants to act without thinking.",
        }
    return None


def check_vague_work_phrases(fields: dict) -> dict | None:
    """Flag posts with vague 'easy money' or 'work from home' pitch language."""
    combined = _combined_text(fields).lower()
    found = [kw for kw in VAGUE_PHRASES if kw in combined]
    if len(found) >= 2:  # require at least 2 vague phrases
        return {
            "type": "warning",
            "text": "The post uses multiple vague phrases like 'earn money online' or 'no experience needed', which are common in scams.",
        }
    if found:
        return {
            "type": "info",
            "text": f"The post uses a phrase like '{found[0]}', which sometimes appears in low-quality or misleading posts.",
        }
    return None


def check_short_description(fields: dict) -> dict | None:
    """Flag very short descriptions — real jobs usually describe the role clearly."""
    desc = fields.get("description", "") or ""
    word_count = len(desc.strip().split())
    if word_count == 0:
        return {
            "type": "warning",
            "text": "No job description was provided. This is unusual for a real job posting.",
        }
    if word_count < MIN_DESCRIPTION_WORDS:
        return {
            "type": "warning",
            "text": f"The job description is very short ({word_count} words). Real job postings usually include detailed responsibilities and requirements.",
        }
    return None


def check_excessive_caps(fields: dict) -> dict | None:
    """Flag posts with many ALL-CAPS words, a common scam pattern."""
    combined = _combined_text(fields)
    words = combined.split()
    caps_words = [w for w in words if w.isupper() and len(w) >= 3]
    if len(caps_words) >= 5:
        return {
            "type": "info",
            "text": f"The post contains {len(caps_words)} ALL-CAPS words, which is unusual for professional job postings.",
        }
    return None


def check_excessive_exclamations(fields: dict) -> dict | None:
    """Flag posts with many exclamation marks."""
    combined = _combined_text(fields)
    count = combined.count("!")
    if count >= 4:
        return {
            "type": "info",
            "text": f"The post has {count} exclamation marks. Excessive enthusiasm can be a sign of low-quality or misleading posts.",
        }
    return None


def check_no_requirements(fields: dict) -> dict | None:
    """Flag when requirements field is blank — common in fake posts."""
    val = fields.get("requirements", "") or ""
    if not val.strip():
        return {
            "type": "info",
            "text": "No requirements were listed. Real jobs usually specify skills or qualifications.",
        }
    return None


def check_telecommuting_no_company(fields: dict) -> dict | None:
    """Flag remote jobs with no company profile — a specific scam pattern."""
    telecommute = str(fields.get("telecommuting", "0")).strip() in ("1", "True", "true")
    profile = fields.get("company_profile", "") or ""
    if telecommute and not profile.strip():
        return {
            "type": "warning",
            "text": "This is a remote job with no company information provided — a pattern common in work-from-home scams.",
        }
    return None


# ---------------------------------------------------------------------------
# Master check function
# ---------------------------------------------------------------------------

ALL_RULES = [
    check_missing_company_profile,
    check_missing_logo,
    check_money_request,
    check_personal_email,
    check_urgency_language,
    check_telecommuting_no_company,
    check_short_description,
    check_vague_work_phrases,
    check_missing_salary,
    check_excessive_caps,
    check_excessive_exclamations,
    check_no_requirements,
]


def run_all_rules(fields: dict) -> List[dict]:
    """
    Run all rule checks on the given job fields.

    Args:
        fields: dict of job posting fields (any/all may be missing).

    Returns:
        List of triggered rule dicts: [{"type": ..., "text": ...}, ...]
        Sorted: warnings first, then info.
    """
    triggered = []
    for rule_fn in ALL_RULES:
        try:
            result = rule_fn(fields)
            if result is not None:
                triggered.append(result)
        except Exception:
            pass  # never let a rule crash the prediction

    # Sort: warnings first
    triggered.sort(key=lambda r: 0 if r["type"] == "warning" else 1)
    return triggered


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _combined_text(fields: dict) -> str:
    """Combine all text fields into one string for keyword checking."""
    text_keys = [
        "title", "company_profile", "description",
        "requirements", "benefits", "location",
    ]
    parts = [str(fields.get(k, "") or "") for k in text_keys]
    return " ".join(parts)

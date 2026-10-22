# Product Requirements Document (PRD)

**Project:** Spotting Fake Job Postings Online
**Course:** ECSCI24302 Machine Learning Essentials
**Team:** Dhanvi Gandhi (2410111028), Saumya Adani (2410111002)
**Timeline:** 1 week (MVP-first)
**Status:** Planning. ML model not started.

---

## 1. Overview

A public website where a user pastes a job posting (or fills a detailed form) and instantly gets a prediction of whether it looks **Real** or **Fake**, along with a plain-language explanation of *why* it was flagged.

The model is trained on the EMSCAD dataset (Kaggle: "Real / Fake Job Posting Prediction"), and the site is built with Streamlit and deployed on Render.

## 2. Problem

Scammers publish fake job posts to steal personal data or collect fake "registration / training / security" fees. Students and fast-moving job seekers are the most likely to be fooled, and a normal person cannot reliably tell a fake post from a real one just by reading it.

## 3. Goals

| # | Goal | How we measure it |
|---|------|-------------------|
| G1 | Detect fake postings accurately despite class imbalance | F1-score on the fake class is the primary metric; also Recall, Precision, ROC-AUC |
| G2 | Make results understandable to non-technical users | Every result shows at least 2 to 3 human-readable reasons |
| G3 | Be usable by anyone in under 30 seconds | Paste, click, see result. No login, no signup |
| G4 | Serve three purposes: course submission, portfolio piece, real public tool | Clean docs, clean repo, live URL, honest disclaimers |
| G5 | Respect user privacy | Nothing the user enters is stored or logged |

## 4. Non-Goals (MVP)

These are intentionally **out of scope** for the 1-week build. They may be listed as "future work".

- Batch CSV upload
- Model performance dashboard with live charts (a simple static metrics summary on the About page is allowed)
- User accounts, history, or saved results
- Scraping job sites or analysing a job URL automatically
- Browser extension or mobile app
- Reporting fake posts to job portals

## 5. Target Users

- **Primary:** Students and fresh graduates searching for jobs and internships.
- **Secondary:** Anyone job-hunting who wants a second opinion on a suspicious post.
- **Tertiary:** Course evaluators and recruiters viewing this as a portfolio project.

## 6. User Stories

1. As a job seeker, I can paste the full text of a job posting and see whether it looks real or fake.
2. As a job seeker, I can see *why* a post was flagged (red flags in plain English).
3. As a careful user, I can fill in individual fields (title, company profile, salary, logo, etc.) for a more detailed check.
4. As a user, I see a confidence level, not just a yes/no, so I know how sure the tool is.
5. As a user, I get safety tips on what to do next if a post looks suspicious.
6. As a user, I am told clearly that my input is not stored.
7. As an evaluator, I can read how the model was trained and how well it performs.

## 7. Functional Requirements

### FR1. Quick Paste mode (default)
- One large text box for the whole job posting.
- Minimum input length check (for example, at least 50 characters) with a friendly message.
- "Check this job" button, plus a "Try an example" button that loads a sample real and a sample fake post.

### FR2. Detailed Form mode
- Fields mirroring the dataset: title, location, department, salary range, company profile, description, requirements, benefits, employment type, required experience, required education, industry, function.
- Toggles: remote/telecommuting, has company logo, has screening questions.
- All fields optional except description (or at least one text field). Missing fields are handled like missing values in training.

### FR3. Prediction result
- Verdict label: **Likely Real**, **Suspicious**, or **Likely Fake**.
- Fake-probability score shown as a percentage and a simple gauge or progress bar.
- Decision threshold comes from validation tuning (see `ML_PIPELINE.md`), not a hard-coded 0.5.

### FR4. Explanation ("Why was this flagged?")
- **Rule-based red flags** (always shown when triggered), for example:
  - No company profile or no company logo
  - No salary information combined with very high promised pay
  - Requests for money, fees, deposits, or bank/ID details
  - Personal email domains (gmail, yahoo, etc.) for contact
  - Very short or very vague description
  - Excessive urgency words ("urgent", "immediate joining", "limited seats")
  - ALL-CAPS or excessive exclamation marks
- **Model-driven reasons:** top words or features pushing the score toward Fake or Real.
- Each reason is written in a plain sentence, never as raw feature names.

### FR5. Safety guidance
- A short "What to do next" section: never pay to get a job, verify the company on its official site, check the recruiter's email domain, be careful sharing ID or bank details.

### FR6. About / How it works page
- Plain explanation of the dataset, models tried, and headline metrics (Recall, Precision, F1, ROC-AUC).
- Honest limitations section.
- Team names and course.

### FR7. Privacy
- No database, no logging of user text. Input is processed in memory only.
- A visible privacy note on the main page.

## 8. Non-Functional Requirements

| Area | Requirement |
|------|-------------|
| Performance | Prediction returns in under 2 seconds once the app is warm |
| Cold start | Render free tier sleeps when idle, so the first load can take 30 to 60 seconds. Document it, and warm the app before any demo |
| Memory | Must run within 512 MB RAM (Render free tier). Keep the model artifact small |
| Reliability | Never crash on empty, huge, or non-English input. Show friendly errors |
| Accessibility | Readable contrast, not colour-only signals (use icons and text with colours), keyboard-friendly |
| Responsiveness | Works on a phone screen, since students will check posts on mobile |
| Reproducibility | Fixed random seed, pinned library versions, saved preprocessing and model pipeline |

## 9. Success Criteria

- Live public URL on Render.
- Works end to end: paste, predict, explain.
- Model beats the baseline clearly on F1 (fake class). Exact target is set after the first training run and recorded in `ML_PIPELINE.md`.
- Repo contains README, docs, training notebook, and a working `requirements.txt`.
- Demo-ready for the course presentation.

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Severe class imbalance, so a lazy model looks accurate | Use class weights, threshold tuning, and report Recall, Precision, F1, never accuracy alone |
| Dataset is old (EMSCAD, around 2014), so modern scam wording may be missed | State this limitation openly; rule-based flags cover common modern patterns |
| False accusations against real companies | Use "Suspicious / Likely Fake" wording, a clear disclaimer, never name or shame companies |
| Render free tier cold starts and 512 MB limit | Small model (TF-IDF + linear or small forest), document cold start, warm up before demos |
| Model file too big for GitHub or Render | Cap TF-IDF features, limit tree count, check file size before committing |
| Only 1 week | Strict MVP scope; stretch features only after deploy works |

## 11. One-Week Milestones

| Day | Deliverable |
|-----|-------------|
| 1 | Repo set up, dataset loaded, EDA done, confirm class counts |
| 2 | Preprocessing and feature pipeline, baseline model |
| 3 | Logistic Regression and Random Forest/XGBoost trained, tuned, compared, best model saved |
| 4 | Explanation logic (rule flags + model reasons), `src/predict.py` working |
| 5 | Streamlit UI built (Quick Paste + Detailed Form + results + About) |
| 6 | Deploy to Render, test on phone and desktop, fix bugs |
| 7 | Polish, write final report/slides, record demo screenshots |

## 12. Open Questions (to resolve during build)

- Final model choice depends on validation results (see decision rule in `ML_PIPELINE.md`).
- Exact verdict thresholds for "Suspicious" versus "Likely Fake" are set after seeing the validation probability distribution.
- Whether to add a second language or a stretch feature (batch CSV) if time remains.

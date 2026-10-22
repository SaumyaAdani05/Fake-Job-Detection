# Tech Stack

**Principle:** Keep it small, Python-only, free to host, and able to run inside Render's free 512 MB limit.

---

## 1. Stack Summary

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11 | Matches the ML libraries, single language for everything |
| Web app / UI | **Streamlit** | Fastest way to build an ML website, no separate frontend or API needed |
| ML | scikit-learn | Logistic Regression, Random Forest, TF-IDF, metrics, pipelines |
| Optional ML | XGBoost | Only if it clearly beats Random Forest and fits in memory |
| Data | pandas, numpy | Loading and cleaning the dataset |
| Model storage | joblib | Save and load the full sklearn Pipeline as one file |
| Plots (EDA/report) | matplotlib, seaborn | Used in notebooks, not required at runtime |
| Notebook | Jupyter / Google Colab | EDA and model training |
| Hosting | **Render** (Web Service, free tier) | Runs the Streamlit app as a long-running server |
| Source control | Git + GitHub | Render deploys directly from the repo |

## 2. Decision Log

| Decision | Reason | Alternative rejected |
|----------|--------|----------------------|
| Streamlit instead of React/Next.js | 1-week deadline, Python-only, built-in widgets | React + FastAPI: more flexible UI but 2 to 3x the work |
| Render only (no Vercel) | Streamlit needs a persistent server. Vercel is serverless and cannot host it | Vercel + Render split only makes sense with a React/Next.js frontend |
| No database | Privacy goal: nothing is stored | Postgres/SQLite: unnecessary and conflicts with the privacy goal |
| No external AI/LLM API | Keeps it free, offline-capable, and about classic ML (the course topic) | LLM-based detection: off-topic for the course and adds cost |
| Single saved Pipeline (`joblib`) | Guarantees training and prediction use identical preprocessing | Separate vectorizer + model files: risk of mismatch |

## 3. Python Libraries (starting `requirements.txt`)

```
streamlit==1.39.0
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
```

Add only if used:
```
xgboost==2.1.1
```

Rules for versions:
- **Pin exact versions.** The scikit-learn version used to train must equal the one used in deployment, otherwise the saved model may fail to load or behave differently.
- Training-only libraries (matplotlib, seaborn, jupyter) go in `requirements-dev.txt`, not in the deployed `requirements.txt`, to keep the Render build small and fast.
- Verify exact versions against what you actually train with, then copy those into `requirements.txt`.

## 4. Project Structure

```
fake-job-detector/
├── app.py                      # Streamlit entry point (Check a Job page)
├── pages/
│   └── 1_How_It_Works.py       # About page
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning, missing-value handling, feature building
│   ├── rules.py                # Rule-based red flag detection
│   ├── explain.py              # Turns model + rules into plain-English reasons
│   └── predict.py              # Loads model, returns label, probability, reasons
├── models/
│   ├── fake_job_model.joblib   # Trained sklearn Pipeline (keep small)
│   └── metrics.json            # Final metrics shown on About page
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── data/                       # GITIGNORED. Holds fake_job_postings.csv locally only
├── assets/
│   └── examples.json           # Sample real and fake posts for the "Try an example" button
├── docs/                       # PRD, UI/UX, tech stack, pipeline, dataset, deployment docs
├── .streamlit/
│   └── config.toml             # Theme and server settings
├── requirements.txt
├── requirements-dev.txt
├── render.yaml                 # Optional Render blueprint
├── .gitignore
├── README.md
├── RULES.md
└── CLAUDE.md
```

## 5. Architecture

```
 User (browser)
      |
      v
 Streamlit app (app.py)  <-- runs on Render
      |
      v
 src/predict.py
   |-- src/preprocess.py  (clean text, build features)
   |-- models/fake_job_model.joblib  (TF-IDF + classifier Pipeline)
   |-- src/rules.py       (rule-based red flags)
   '-- src/explain.py     (plain-English reasons)
      |
      v
 Verdict + probability + reasons  -> rendered in the UI
```

- The model is loaded **once** using `@st.cache_resource` so it is not reloaded on every click.
- No network calls, no database, no user data written to disk.

## 5.1 Interface between UI and ML code

`src/predict.py` exposes one function the UI calls:

```python
def predict_job(fields: dict) -> dict:
    """
    fields: dict of dataset-style inputs (title, description, company_profile, ...).
            Missing or blank fields are allowed.
    returns: {
        "label": "Likely Fake" | "Suspicious" | "Likely Real",
        "fake_probability": float,      # 0.0 to 1.0
        "reasons": [ {"type": "warning"|"info"|"ok", "text": str}, ... ],
    }
    """
```

Quick Paste mode puts the pasted text into `description` and leaves other fields empty (or splits obvious sections if simple to do).

## 6. ML Stack Details

- **Text features:** TF-IDF on combined text fields (title + company_profile + description + requirements + benefits). Cap `max_features` (for example 5,000 to 20,000) and use n-grams (1,2) only if memory allows.
- **Categorical features:** employment_type, required_experience, required_education, industry, function (one-hot, with rare categories grouped).
- **Binary and engineered features:** telecommuting, has_company_logo, has_questions, missing salary, missing company profile, text length, count of "!" and ALL-CAPS words, personal-email present, etc.
- **Imbalance handling:** `class_weight="balanced"` and threshold tuning. SMOTE only inside cross-validation folds, if used at all.
- Details are in `ML_PIPELINE.md`.

## 7. Deployment Stack (Render)

| Setting | Value |
|---------|-------|
| Service type | Web Service |
| Runtime | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Instance | Free tier (512 MB RAM, sleeps after about 15 minutes idle) |
| Env var | `PYTHON_VERSION=3.11.9` |
| Auto-deploy | On push to `main` |

Full steps are in `DEPLOYMENT.md`.

## 8. Tooling

- **Editor:** VS Code (or Cursor / Claude Code).
- **Linting/format (optional):** `ruff` and `black` in dev only.
- **Testing (light):** a few `pytest` tests for `rules.py` and `predict.py` (empty input, long input, obvious fake, obvious real).

## 9. Constraints to Respect

- Render free tier: **512 MB RAM**, slow cold starts, no persistent disk.
- GitHub file limit: **100 MB per file** (keep the model well under 50 MB).
- No secrets are needed for this project. If any are ever added, use Render environment variables, never commit them.

## 10. Stretch Options (only after MVP is live)

- Batch CSV upload tab.
- Static model comparison charts on the About page.
- Hugging Face Spaces mirror (avoids Render cold-start sleep).
- Next.js frontend + FastAPI backend (would justify using Vercel).

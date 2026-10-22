# Fake Job Post Checker

Paste a job posting and find out whether it looks **Real**, **Suspicious**, or **Likely Fake**, with plain-English reasons.

**Course:** ECSCI24302 Machine Learning Essentials
**Team:** Dhanvi Gandhi (2410111028), Saumya Adani (2410111002)
**Live app:** _add Render URL here after deployment_
**Note:** The free hosting tier sleeps when idle, so the first load may take up to a minute.

---

## What it does

Scammers post fake jobs to steal personal data or collect fake fees. This tool uses a machine learning model trained on about 17,880 labelled job postings to estimate how likely a post is to be fake, and explains which red flags it noticed.

- **Quick Paste:** paste the full job post and click check.
- **Detailed Form:** fill in individual fields (title, company profile, salary, logo, etc.) for a more careful check.
- **Why flagged:** shows rule-based red flags and the words that influenced the model.
- **Private:** your text is processed in memory and is never stored.

> This tool gives an automated estimate, not proof. Always verify a company through its official website.

## Dataset

Employment Scam Aegean Dataset (EMSCAD), shared on Kaggle as "Real / Fake Job Posting Prediction" by Shivam Bansal.
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
The dataset is not included in this repository. Download `fake_job_postings.csv` into the `data/` folder to retrain. See `DATASET.md`.

## Models

1. Baseline (dummy classifier)
2. Logistic Regression with engineered clues (missing salary, logo, company profile, etc.)
3. Random Forest / XGBoost

Because fake posts are rare, models are compared mainly on **F1-score, Recall, and Precision** (plus Confusion Matrix, ROC-AUC, PR-AUC), not accuracy. See `ML_PIPELINE.md`.

_Final results: add the metrics table here after training._

## Tech stack

Python 3.11, Streamlit, scikit-learn, pandas, joblib. Hosted on Render. See `TECH_STACK.md`.

## Run locally

```bash
git clone <your-repo-url>
cd fake-job-detector
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

```bash
pip install -r requirements-dev.txt
# put fake_job_postings.csv in data/
jupyter notebook notebooks/02_modeling.ipynb
```
The notebook saves `models/fake_job_model.joblib` and `models/metrics.json`.

## Project structure

```
app.py                 Streamlit entry point
pages/                 Extra pages (How It Works)
src/                   preprocessing, rules, explanations, prediction
models/                trained model + metrics
notebooks/             EDA and modelling
assets/                example posts
docs/                  PRD.md, UI_UX.md, TECH_STACK.md, ML_PIPELINE.md, DATASET.md, DEPLOYMENT.md
RULES.md, CLAUDE.md    project rules and AI-assistant instructions
```

(Move `PRD.md`, `UI_UX.md`, `TECH_STACK.md`, `ML_PIPELINE.md`, `DATASET.md`, and `DEPLOYMENT.md` into `docs/` when you set up the repo.)

## Deployment

Deployed as a Render Web Service. Steps are in `DEPLOYMENT.md`.

## Limitations

- Trained on an older dataset (around 2012 to 2014), so newer scam styles may be missed.
- Mostly English postings.
- Predictions are probabilities and can be wrong in both directions.

## Documentation index

| File | Purpose |
|------|---------|
| `PRD.md` | Requirements, scope, milestones |
| `UI_UX.md` | Design and user experience |
| `TECH_STACK.md` | Technology choices and structure |
| `ML_PIPELINE.md` | Training and evaluation plan |
| `DATASET.md` | Dataset details and caveats |
| `DEPLOYMENT.md` | Render deployment guide |
| `RULES.md` | Do's and don'ts |
| `CLAUDE.md` | Instructions for AI coding tools |

## Credits

Dataset: EMSCAD, University of the Aegean (via Kaggle, Shivam Bansal). Built by Dhanvi Gandhi and Saumya Adani.

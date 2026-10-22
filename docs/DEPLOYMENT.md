# Deployment Guide (Render + Streamlit)

**Target:** Render Web Service, free tier, deployed from GitHub.
**Why Render only:** Streamlit needs a long-running server, so it cannot run on Vercel's serverless platform.

---

## 1. Pre-Deployment Checklist

- [ ] Model trained, saved as `models/fake_job_model.joblib`, and **under 50 MB**
- [ ] `models/metrics.json` present
- [ ] `requirements.txt` has **pinned versions**, and scikit-learn version equals the one used for training
- [ ] `data/` is in `.gitignore` (dataset is not pushed)
- [ ] App runs locally: `streamlit run app.py`
- [ ] No hard-coded absolute paths (use paths relative to the project root)
- [ ] App loads model via `@st.cache_resource`
- [ ] No secrets or API keys in the repo
- [ ] README has the live URL placeholder

## 2. Local Test First

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Test: empty input, short input, long input, one fake-like example, one real-like example, both tabs, phone-width browser.

## 3. Prepare the Repo

`.gitignore` should include:
```
data/
.venv/
__pycache__/
*.ipynb_checkpoints
.DS_Store
.env
```

Push to GitHub (public or private, Render can access both after you connect your account).

## 4. Create the Render Service

1. Create an account at render.com and connect your GitHub.
2. Click **New +** then **Web Service** and pick the repository.
3. Settings:

| Setting | Value |
|---------|-------|
| Name | `fake-job-detector` (or similar) |
| Region | Closest to your users (for example Singapore) |
| Branch | `main` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Instance Type | Free |

4. Add environment variable: `PYTHON_VERSION` = `3.11.9`.
5. Click **Create Web Service** and watch the build logs.

## 5. Optional: `render.yaml` (Blueprint)

```yaml
services:
  - type: web
    name: fake-job-detector
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
```

## 6. `.streamlit/config.toml`

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true

[theme]
primaryColor = "#1D4ED8"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0F172A"
font = "sans serif"

[browser]
gatherUsageStats = false
```

## 7. Post-Deployment Checks

- [ ] Open the live URL on desktop and phone
- [ ] Run a fake-like and a real-like example
- [ ] Confirm results match local results
- [ ] Check memory use in the Render dashboard (stay well below 512 MB)
- [ ] Confirm nothing is written to disk or logged from user input
- [ ] Add the live URL to README and the report

## 8. Free-Tier Behaviour to Know

- The service **sleeps after about 15 minutes of no traffic**. The next visit can take **30 to 60+ seconds** to wake up.
- **Before a demo or viva, open the URL a few minutes earlier** to warm it up.
- Free instances have limited monthly hours and no persistent disk. This app needs neither.
- Free-tier limits and policies change, so double-check Render's current pricing page.

## 9. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| Build fails on a package | Version conflict or too heavy | Check pinned versions, move training-only libs to `requirements-dev.txt` |
| App crashes on start, "Out of memory" | Model or vectorizer too large | Reduce `max_features`, fewer trees, `compress=3` in joblib, or prefer Logistic Regression |
| `ValueError` or unpickle error loading model | scikit-learn version mismatch | Retrain or install the same version used in training |
| Port or "no open ports detected" | Wrong start command | Use `--server.port $PORT --server.address 0.0.0.0` |
| Model file not found | Path issue or file not committed | Use relative paths; check the file is in Git (and below GitHub's 100 MB limit) |
| Very slow first load | Cold start | Warm the app before demos |

## 10. Updating the Live Site

Commit and push to `main`. Render auto-deploys. If you retrain the model, re-save the model and `metrics.json`, commit them, and push.

## 11. Fallback Hosting (if Render is a problem)

- **Hugging Face Spaces** (Streamlit or Docker): free, good for ML demos, no 15-minute sleep behaviour on the same terms.
- **Streamlit Community Cloud**: free, deploys from GitHub in a few clicks.

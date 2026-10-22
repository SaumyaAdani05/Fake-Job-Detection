# Project Rules: What To Do and What Not To Do

Everyone working on this project (including AI coding tools) follows these rules.

---

## 1. Scope Rules

**DO**
- Build the MVP first: Quick Paste + Detailed Form + verdict + reasons + About page.
- Finish and **deploy** before adding anything extra.
- Check each new idea against `PRD.md`. If it is in "Non-Goals", it waits.

**DON'T**
- Don't add batch CSV upload, dashboards, accounts, history, or scraping during the MVP.
- Don't add new frameworks (React, FastAPI, databases) without a team decision and an update to `TECH_STACK.md`.
- Don't spend more than the planned time on one day's task. Cut scope instead.

## 2. Machine Learning Rules

**DO**
- Split data first (stratified, seed 42), then fit preprocessing on the training split only.
- Judge models on **F1 (Fake class), Recall, Precision, PR-AUC, ROC-AUC, confusion matrix**.
- Keep the whole thing inside a single scikit-learn `Pipeline`.
- Tune the decision threshold on validation data.
- Save the final test result in `models/metrics.json`.
- Write a short error analysis (what kinds of fake posts are missed).

**DON'T**
- Don't use accuracy as the headline metric.
- Don't tune anything on the test set.
- Don't apply SMOTE/oversampling before the split.
- Don't use `job_id` as a feature.
- Don't copy a model from the internet. Train and understand your own.
- Don't commit a model larger than 50 MB.

## 3. Data and Privacy Rules

**DO**
- Process user input in memory only.
- Show a visible "Your text is not saved" note.
- Keep `data/` out of Git.
- Write your own example posts for the demo button.

**DON'T**
- Don't store, log, print, or send user input anywhere (no database, no analytics of text, no third-party APIs).
- Don't push the raw dataset to GitHub or show raw dataset rows on the site.
- Don't ask the user for personal information.
- Don't name real companies as scammers.

## 4. Wording and Ethics Rules

**DO**
- Use "Likely Real / Suspicious / Likely Fake".
- Always show reasons and a disclaimer: *"This is an automated estimate, not proof. Verify the company independently."*
- Be honest about limitations (old dataset, English, not perfect).

**DON'T**
- Don't say "100% safe", "guaranteed scam", or "verified".
- Don't tell users a real company is definitely a fraud.
- Don't overstate accuracy in the report or on the site.

## 5. UI Rules

**DO**
- Follow `UI_UX.md` (clean, trustworthy, light theme, plain language).
- Pair colour with icon and text.
- Make sure it works on a phone.

**DON'T**
- Don't expose model internals (coefficients, raw tokens) to end users.
- Don't show raw decimals like `0.8731`. Use "87%".
- Don't add pop-ups, ads, or sign-up walls.

## 6. Code Rules

**DO**
- Keep logic in `src/`, UI in `app.py` and `pages/`.
- Use clear function names, type hints, and short docstrings.
- Pin library versions in `requirements.txt`.
- Handle empty, short, very long, and odd-character input without crashing.
- Use `@st.cache_resource` to load the model once.
- Use relative paths.
- Write a few `pytest` tests for `rules.py` and `predict.py`.

**DON'T**
- Don't put training code inside `app.py`.
- Don't hard-code absolute paths or machine-specific values.
- Don't commit secrets, `.env`, virtual environments, or the dataset.
- Don't leave debug prints in the final app.
- Don't change preprocessing in the app unless training is changed the same way.

## 7. Git Rules

**DO**
- Commit small and often, with clear messages (for example `Add rule-based red flag checks`).
- Use branches for bigger changes, merge to `main` when it works.
- Pull before you push when two people are working.

**DON'T**
- Don't push broken code to `main` (Render auto-deploys from it).
- Don't commit large files or notebook outputs full of data dumps.

## 8. Deployment Rules

**DO**
- Test locally, then deploy.
- Warm the Render app before any demo.
- Re-test the live URL after each deployment.

**DON'T**
- Don't deploy with unpinned dependencies.
- Don't assume Vercel works for Streamlit (it does not).
- Don't ignore memory warnings on Render.

## 9. Team Rules

- Decide ownership early. Suggested split: one person leads ML (EDA, models, evaluation), the other leads app (Streamlit UI, deployment), and both review each other's work and write the report together.
- Both team members must be able to explain the whole pipeline in the viva.
- Update the docs if a decision changes.

## 10. Definition of Done

The project is done when:
- [ ] Live URL works on desktop and phone
- [ ] Quick Paste and Detailed Form both work
- [ ] Verdict, probability, and reasons display correctly
- [ ] Privacy note and disclaimer are visible
- [ ] Metrics and limitations are on the About page
- [ ] README is complete
- [ ] Notebooks run top to bottom without errors
- [ ] Report and slides match the real results

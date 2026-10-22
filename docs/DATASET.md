# Dataset Notes

## 1. Source

- **Name:** Employment Scam Aegean Dataset (EMSCAD)
- **Kaggle title:** "Real / Fake Job Posting Prediction" by Shivam Bansal
- **Link:** https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- **File:** `fake_job_postings.csv`
- **Original creators:** Laboratory of Information & Communication Systems Security, University of the Aegean (Greece)

Check the Kaggle page for the current licence and citation requirements, and credit the dataset in the report and on the About page.

## 2. Size

- About **17,880 job postings**.
- Proposal states 18 "features". The CSV has 18 columns in total, and one of them is the target (`fraudulent`) and one is an ID (`job_id`), so there are roughly 16 usable input columns. Verify and state this precisely in the report.

## 3. Columns (expected)

| Column | Type | Notes |
|--------|------|-------|
| `job_id` | int | Identifier. **Drop.** |
| `title` | text | Job title |
| `location` | text | Format like `US, NY, New York`. Extract country |
| `department` | text | Very sparse |
| `salary_range` | text | Mostly missing. Missingness is a signal |
| `company_profile` | text | Often missing in fake posts |
| `description` | text | Main text signal |
| `requirements` | text | Text signal |
| `benefits` | text | Text signal |
| `telecommuting` | 0/1 | Remote flag |
| `has_company_logo` | 0/1 | Fake posts less likely to have one |
| `has_questions` | 0/1 | Screening questions present |
| `employment_type` | categorical | Full-time, Part-time, Contract, etc. |
| `required_experience` | categorical | Entry level, Mid-Senior, etc. |
| `required_education` | categorical | Bachelor's, High School, etc. |
| `industry` | categorical | Many categories |
| `function` | categorical | Many categories |
| `fraudulent` | 0/1 | **Target.** 1 = Fake |

## 4. IMPORTANT: Verify the Class Counts

The proposal says about **16,244 real and 1,636 fake** (about 1 in 11 fake). Published descriptions of this dataset commonly report about **17,014 real and 866 fake** (about 4.8% fake, roughly 1 in 21).

**Action on Day 1:** run `df["fraudulent"].value_counts()` and use the real numbers in all documents, the report, and the About page. If the counts differ from the proposal, update the proposal statement ("about 1 in 11") to match. The imbalance is probably *more* severe than the proposal says, which makes F1/Recall/PR-AUC even more important than accuracy.

## 5. Known Data Issues

- **Severe class imbalance** (fake is the minority).
- **Many missing values**, especially `salary_range`, `department`, `company_profile`, `benefits`.
- **Duplicate and near-duplicate postings.** Remove exact duplicates before splitting.
- **Messy text:** HTML fragments, extra whitespace, odd characters, mixed capitalisation.
- **Age:** data was collected around 2012 to 2014. Scam wording has evolved (for example Telegram/WhatsApp-based tasks, crypto, "task scams"). The model may miss modern patterns.
- **Mostly English** postings, many from the US. Not representative of Indian job platforms.
- Possible **dataset-specific artefacts**: some fake posts may share templates, which can inflate scores. Mention this in limitations.

## 6. Handling Rules

- Store the CSV only in `data/` and keep it **out of Git** (add `data/` to `.gitignore`). Do not redistribute the data from the repo or the website.
- Never show raw dataset postings on the live website. Write your own example posts for the "Try an example" button (`assets/examples.json`), based on common patterns, with no real company names.
- Do not name or accuse real companies found in the dataset.

## 7. Day-1 Checklist

- [ ] Download dataset from Kaggle
- [ ] Load and print shape, dtypes
- [ ] Print real vs fake counts and percentage (update docs if different)
- [ ] Count duplicates
- [ ] Missing value table
- [ ] Save a short EDA summary for the report

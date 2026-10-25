
import streamlit as st

st.set_page_config(
    page_title="How It Works | Fake Job Detection",
    page_icon="📘",
    layout="wide",
)

# Professional styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F7F9FC;
        }

        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #000000 !important;
        }

        p, label, .stMarkdown, .stMarkdown p {
            color: #000000 !important;
        }

        .info-card {
            background: #FFFFFF;
            border: 1px solid #DCE4EE;
            border-radius: 12px;
            padding: 20px;
            height: 100%;
        }

        .metric-value {
            color: #000000 !important;
            font-size: 28px;
            font-weight: 700;
        }

        .metric-label {
            color: #334155 !important;
            font-size: 14px;
            font-weight: 500;
        }

        .step-number {
            color: #17365D;
            font-size: 24px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page header
st.markdown(
    "<p style='color:#64748B;font-size:0.85rem;"
    "font-weight:600;letter-spacing:2px;'>"
    "PROJECT GUIDE</p>",
    unsafe_allow_html=True,
)

st.title("📘 How It Works")
st.markdown(
    "### Understanding the Fake Job Detection system"
)
st.write(
    "This application analyses job-posting information using a "
    "trained machine learning model and rule-based explanations "
    "to help users identify potentially fraudulent vacancies."
)

st.divider()

# Overview
st.header("1. How the system works")

steps = [
    (
        "01",
        "Enter job details",
        "Enter job title, location, company information, description, "
        "requirements and other available details. Alternatively, "
        "paste the complete posting using Quick Paste.",
    ),
    (
        "02",
        "Analyse the posting",
        "The application passes the supplied information to the "
        "trained prediction pipeline, which processes the available "
        "features and estimates the likelihood of fraud.",
    ),
    (
        "03",
        "Review the result",
        "The application displays a verdict, estimated fake probability "
        "and explanatory warning signs or positive indicators.",
    ),
]

cols = st.columns(3)

for col, (number, title, description) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="step-number">{number}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# Model information
st.header("2. Machine learning model")

st.markdown(
    """
    The project uses **tuned Logistic Regression** with a custom
    text and rule-based preprocessing pipeline.

    Logistic Regression estimates the probability of a posting
    belonging to the fraudulent class based on patterns learned
    during training. The application combines the model output
    with rule-based explanations to help users understand possible
    warning signs.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>Model</h3>
            <p><b>Algorithm:</b> Tuned Logistic Regression</p>
            <p><b>Input:</b> Job-posting text and supported fields</p>
            <p><b>Output:</b> Risk label, fake probability and reasons</p>
            <p><b>Decision threshold:</b> 0.7972</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Prediction labels</h3>
            <p>🚨 <b>Likely Fake:</b> Stronger indicators of fraud.</p>
            <p>⚠️ <b>Suspicious:</b> Further verification is advisable.</p>
            <p>✅ <b>Likely Real:</b> Fewer indicators of fraud were detected.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "The labels reflect the application's prediction pipeline. "
    "They are not definitive proof of whether an employer is genuine."
)

st.divider()

# Evaluation metrics
st.header("3. Model evaluation")

st.write(
    "The following figures are the reported test-set metrics "
    "provided for this project's trained model."
)

metrics = [
    ("84.17%", "Fake-class precision"),
    ("78.91%", "Fake-class recall"),
    ("81.45%", "Fake-class F1-score"),
    ("98.88%", "ROC-AUC"),
]

cols = st.columns(4)

for col, (value, label) in zip(cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("What do these metrics mean?"):
    st.markdown(
        """
        - **Precision (84.17%):** Among postings predicted as fake,
          the proportion that were actually fake in the evaluation set.
        - **Recall (78.91%):** The proportion of fake postings in the
          evaluation set that the model successfully identified.
        - **F1-score (81.45%):** A combined measure of precision and
          recall.
        - **ROC-AUC (98.88%):** Measures how well the model separates
          fake and real postings across different classification
          thresholds.
        """
    )

st.info(
    "These are test-set results, not a guarantee of performance on "
    "every new job posting. Results may vary with the data and context."
)

st.divider()

# Rules and safety
st.header("4. Warning signs the application may highlight")

st.markdown(
    """
    Depending on the information supplied, the explanation system
    may flag indicators such as:

    - Requests for registration fees or payments from applicants.
    - Missing company information or a company logo.
    - Pressure language that urges applicants to act immediately.
    - Very short or incomplete job descriptions.
    - Other patterns identified by the project's rules and model.

    A single indicator does not prove that a posting is fraudulent.
    Likewise, a detailed description or company logo does not prove
    that an employer is legitimate.
    """
)

st.divider()

# Responsible use
st.header("5. How to stay safe when applying")

st.markdown(
    """
    1. Verify the employer through its official website and contact details.
    2. Never pay a recruitment or registration fee to obtain a job.
    3. Be cautious about requests for bank details, passwords or identity
       documents before verifying the employer.
    4. Check whether the vacancy is published through a trustworthy
       recruitment channel.
    5. Use the model as a screening aid, not as the only basis for a decision.
    """
)

st.warning(
    "This project is an academic screening tool. Its predictions are "
    "not a substitute for independent verification or professional advice."
)

st.divider()

st.caption(
    "Academic project | Fake Job Detection | Machine Learning Essentials"
)

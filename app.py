import streamlit as st
from src.predict import predict_job

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔎",
    layout="wide",
)

# Premium SaaS visual system: presentation only; prediction pipeline is unchanged.
st.markdown(
    """
    <style>
    :root {
        --navy: #122744;
        --blue: #2563EB;
        --blue-soft: #EFF6FF;
        --ink: #24364B;
        --muted: #64748B;
        --line: #DCE5F0;
        --surface: #FFFFFF;
        --canvas: #F5F8FC;
    }
    .stApp { background: var(--canvas); }
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 { color: var(--navy) !important; letter-spacing: -0.025em; }
    p, label, .stMarkdown { color: var(--ink); }
    .eyebrow {
        color: var(--blue); font-size: .76rem; font-weight: 800;
        letter-spacing: .15em; text-transform: uppercase; margin-bottom: .35rem;
    }
    .hero {
        background: linear-gradient(115deg, #FFFFFF 0%, #F0F6FF 100%);
        border: 1px solid var(--line); border-radius: 22px;
        padding: 1.8rem 2rem; margin: .4rem 0 1.2rem 0;
        box-shadow: 0 8px 28px rgba(18,39,68,.045);
    }
    .hero-title { color: var(--navy); font-size: 2.15rem; font-weight: 800; line-height: 1.15; }
    .hero-copy { color: #52657D; font-size: 1rem; margin-top: .65rem; max-width: 760px; }
    .pill {
        display: inline-block; background: #E7F0FF; color: #174EA6;
        border: 1px solid #CFE0FF; padding: .28rem .65rem;
        border-radius: 999px; font-size: .75rem; font-weight: 700; margin-bottom: .8rem;
    }
    .section-kicker { color: var(--muted); font-size: .8rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }
    .panel {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 16px; padding: 1.15rem 1.25rem;
        box-shadow: 0 4px 16px rgba(18,39,68,.035); margin-bottom: .6rem;
    }
    .sample-card {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px;
        padding: .9rem 1rem; min-height: 112px;
    }
    .sample-label { color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
    .sample-title { color: var(--navy); font-weight: 750; font-size: 1rem; margin: .3rem 0; }
    .sample-copy { color: #52657D; font-size: .86rem; }
    div[data-testid="stForm"] {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 16px;
        padding: 1.25rem 1.35rem; box-shadow: 0 4px 16px rgba(18,39,68,.035);
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 4px 16px rgba(18,39,68,.035);
    }
    div[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 650; }
    div.stButton > button, div.stFormSubmitButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--blue) !important; color: #FFFFFF !important;
        border: 1px solid var(--blue) !important; border-radius: 10px;
        min-height: 46px; font-weight: 750; transition: all .15s ease;
    }
    div.stButton > button p, div.stFormSubmitButton > button p,
    div[data-testid="stFormSubmitButton"] > button p { color: #FFFFFF !important; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1D4ED8 !important; border-color: #1D4ED8 !important;
        box-shadow: 0 5px 14px rgba(37,99,235,.18);
    }
    div[data-testid="stProgressBar"] { margin: .5rem 0 1rem 0; }
    div[data-testid="stProgressBar"] > div > div { background-color: var(--blue); }
    div[data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: var(--line); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="pill">JOB SAFETY TOOLKIT · ML-POWERED SCREENING</div>
      <div class="hero-title">Know the signs before you apply.</div>
      <div class="hero-copy">
        Review a job posting for potential fraud indicators. Get a risk estimate
        and clear explanations to help you decide what to verify next.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Example scenarios use the same UI fields and prediction function as manual input.
samples = {
    "01 · Registration-fee scam": {
        "title": "Work From Home Data Entry",
        "company_name": "",
        "location": "Remote",
        "employment_type": "Full-time",
        "salary_range": "₹50,000 per week",
        "company_profile": "",
        "description": (
            "Earn high income from home. No experience required. "
            "Pay a registration fee to start immediately. Limited vacancies available."
        ),
        "requirements": "Basic typing skills",
        "benefits": "",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "summary": "An upfront payment request, unusually high pay, and limited employer details.",
        "risk_tag": "HIGH-RISK EXAMPLE",
        "paste": (
            "Work From Home Data Entry\n"
            "Location: Remote\nSalary: ₹50,000 per week\n"
            "No experience required. Earn high income from home. "
            "Pay a registration fee to start immediately. Limited vacancies available."
        ),
    },
    "02 · Vague high-earnings offer": {
        "title": "Online Assistant — Immediate Hiring",
        "company_name": "",
        "location": "Remote",
        "employment_type": "Not specified",
        "salary_range": "₹8,000 per day",
        "company_profile": "",
        "description": (
            "Easy online work. Earn ₹8,000 every day with no experience. "
            "Start today. Message us immediately to reserve your position. "
            "Full details will be shared after selection."
        ),
        "requirements": "",
        "benefits": "Flexible hours",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "summary": "Vague duties, unusually high earnings, and pressure to respond quickly.",
        "risk_tag": "MIXED / SUSPICIOUS SIGNALS",
        "paste": (
            "Online Assistant — Immediate Hiring\nLocation: Remote\n"
            "Salary: ₹8,000 per day\nEasy online work. Earn ₹8,000 every day with no experience. "
            "Start today. Message us immediately to reserve your position. "
            "Full details will be shared after selection."
        ),
    },
    "03 · Detailed software engineer role": {
        "title": "Software Engineer",
        "company_name": "Example Technology Ltd.",
        "location": "Ahmedabad, India",
        "employment_type": "Full-time",
        "salary_range": "₹6–10 LPA",
        "company_profile": (
            "Example Technology Ltd. develops business software for enterprise customers."
        ),
        "description": (
            "We are seeking a Software Engineer to develop, test, and maintain applications. "
            "Responsibilities include writing maintainable code, reviewing code, fixing bugs, "
            "and collaborating with the product team."
        ),
        "requirements": (
            "Bachelor's degree in Computer Science or related field. "
            "Experience with Python, SQL, Git, and software testing."
        ),
        "benefits": "Health insurance and paid leave",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1,
        "summary": "A detailed role with company context, defined responsibilities, and relevant skills.",
        "risk_tag": "LOWER-RISK EXAMPLE",
        "paste": (
            "Software Engineer — Example Technology Ltd.\nLocation: Ahmedabad, India\n"
            "Employment: Full-time\nSalary: ₹6–10 LPA\n"
            "Company: Example Technology Ltd. develops business software for enterprise customers.\n"
            "We are seeking a Software Engineer to develop, test, and maintain applications. "
            "Responsibilities include writing maintainable code, reviewing code, fixing bugs, "
            "and collaborating with the product team.\n"
            "Requirements: Bachelor's degree in Computer Science or related field; Python, SQL, Git, "
            "and software testing. Benefits include health insurance and paid leave."
        ),
    },
    "04 · Incomplete employer details": {
        "title": "Customer Support Executive",
        "company_name": "Bright Future Services",
        "location": "India / Work from home",
        "employment_type": "Full-time",
        "salary_range": "Attractive salary",
        "company_profile": "We are a growing team offering opportunities to motivated candidates.",
        "description": (
            "Hiring customer support executives for remote work. Candidates should be good at "
            "communication and available to start soon. Apply with your details for more information."
        ),
        "requirements": "Good communication skills",
        "benefits": "",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 0,
        "summary": "Some employer information is provided, but pay and job requirements remain unclear.",
        "risk_tag": "CHECK THE DETAILS",
        "paste": (
            "Customer Support Executive — Bright Future Services\n"
            "Location: India / Work from home\nSalary: Attractive salary\n"
            "We are a growing team offering opportunities to motivated candidates. Hiring customer "
            "support executives for remote work. Candidates should be good at communication and "
            "available to start soon. Apply with your details for more information."
        ),
    },
}

def load_sample():
    choice = st.session_state.get("sample_choice", "None — enter my own job")
    sample = samples.get(choice)
    if not sample:
        return
    st.session_state.job_title = sample["title"]
    st.session_state.company_name = sample["company_name"]
    st.session_state.location = sample["location"]
    st.session_state.employment_type = sample["employment_type"]
    st.session_state.salary = sample["salary_range"]
    st.session_state.company_profile = sample["company_profile"]
    st.session_state.requirements = sample["requirements"]
    st.session_state.benefits = sample["benefits"]
    st.session_state.description = sample["description"]
    st.session_state.is_remote = bool(sample["telecommuting"])
    st.session_state.has_logo = bool(sample["has_company_logo"])
    st.session_state.has_questions = bool(sample["has_questions"])
    st.session_state.quick_paste_text = sample["paste"]

st.markdown('<div class="section-kicker">START WITH AN EXAMPLE</div>', unsafe_allow_html=True)
st.markdown("### Explore sample job postings")
st.write("Choose a scenario to populate the form, then run the model to see its assessment.")

sample_choice = st.selectbox(
    "Select an example",
    ["None — enter my own job", *samples.keys()],
    key="sample_choice",
    on_change=load_sample,
)

if sample_choice in samples:
    chosen = samples[sample_choice]
    st.markdown(
        f"""
        <div class="sample-card">
          <div class="sample-label">{chosen["risk_tag"]}</div>
          <div class="sample-title">{sample_choice.split(" · ", 1)[1]}</div>
          <div class="sample-copy">{chosen["summary"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("These are illustrative examples for demonstrating the tool. The model generates the actual result; the example label is not a prediction.")

st.divider()

st.markdown('<div class="section-kicker">JOB ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("### Add the job posting")
mode = st.radio(
    "Choose input method",
    ["Detailed Form", "Quick Paste"],
    horizontal=True,
    label_visibility="collapsed",
)

input_data = {}

if mode == "Quick Paste":
    pasted_text = st.text_area(
        "Paste the complete job posting",
        placeholder=(
            "Paste the job title, company information, description, requirements, "
            "salary, and other details..."
        ),
        height=260,
        key="quick_paste_text",
    )
    submitted = st.button("Analyse job posting", type="primary", use_container_width=True)
    if submitted:
        if not pasted_text.strip():
            st.error("Please paste a job posting first.")
        else:
            input_data = {"description": pasted_text}
else:
    st.markdown("Fill in the details you have. Fields marked * are required.")
    with st.form("job_details_form"):
        st.markdown("#### Role and employer")
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input(
                "Job Title *", placeholder="e.g. Software Engineer", key="job_title"
            )
            company_name = st.text_input(
                "Company Name", placeholder="Enter company name", key="company_name"
            )
            location = st.text_input(
                "Job Location", placeholder="e.g. Ahmedabad, India", key="location"
            )
            employment_type = st.selectbox(
                "Employment Type",
                ["Not specified", "Full-time", "Part-time", "Contract", "Temporary", "Internship"],
                key="employment_type",
            )
            salary = st.text_input(
                "Salary / Compensation", placeholder="e.g. ₹5–8 LPA", key="salary"
            )
        with col2:
            company_profile = st.text_area(
                "Company Profile", placeholder="Describe the company...", height=120,
                key="company_profile",
            )
            requirements = st.text_area(
                "Job Requirements", placeholder="Skills, qualifications and experience...",
                height=120, key="requirements",
            )
            benefits = st.text_area(
                "Benefits", placeholder="Health insurance, leave, etc.", height=80,
                key="benefits",
            )
        description = st.text_area(
            "Job Description *", placeholder="Describe responsibilities and duties...",
            height=180, key="description",
        )
        st.markdown("#### Additional indicators")
        col3, col4, col5 = st.columns(3)
        with col3:
            is_remote = st.checkbox("Remote / telecommuting", key="is_remote")
        with col4:
            has_logo = st.checkbox("Company logo available", key="has_logo")
        with col5:
            has_questions = st.checkbox("Screening questions included", key="has_questions")
        submitted = st.form_submit_button(
            "Analyse job posting", type="primary", use_container_width=True
        )

    if submitted:
        if not job_title.strip() or not description.strip():
            st.error("Please enter both the Job Title and Job Description.")
        else:
            input_data = {
                "title": job_title,
                "company_name": company_name,
                "location": location,
                "salary_range": salary,
                "company_profile": company_profile,
                "description": description,
                "requirements": requirements,
                "benefits": benefits,
                "telecommuting": int(is_remote),
                "has_company_logo": int(has_logo),
                "has_questions": int(has_questions),
                "employment_type": "" if employment_type == "Not specified" else employment_type,
            }

if input_data:
    st.divider()
    st.markdown('<div class="section-kicker">ASSESSMENT RESULTS</div>', unsafe_allow_html=True)
    st.markdown("### Job risk assessment")
    st.caption("Model-generated screening result. Review the explanations and verify the employer independently.")

    try:
        with st.spinner("Analysing the job posting..."):
            result = predict_job(input_data)

        label = result["label"]
        probability = max(0.0, min(1.0, float(result["fake_probability"])))
        if label == "Likely Fake":
            verdict_colour = "#B42318"
            verdict_bg = "#FFF1F0"
            verdict_description = "Several indicators suggest this posting may be fraudulent."
            verdict_icon = "🚩"
        elif label == "Suspicious":
            verdict_colour = "#B54708"
            verdict_bg = "#FFFAEB"
            verdict_description = "Review the warning signs carefully before proceeding."
            verdict_icon = "⚠️"
        else:
            verdict_colour = "#067647"
            verdict_bg = "#ECFDF3"
            verdict_description = "Fewer fraud indicators were detected. This does not verify the employer."
            verdict_icon = "✓"

        st.markdown(
            f"""
            <div style="background:{verdict_bg}; border:1px solid {verdict_colour}33;
                        border-left:5px solid {verdict_colour}; border-radius:16px;
                        padding:1.25rem 1.5rem; margin:.6rem 0 1rem 0;">
              <div style="font-size:.75rem; letter-spacing:.12em; font-weight:800;
                          color:{verdict_colour};">MODEL VERDICT</div>
              <div style="font-size:1.8rem; font-weight:800; color:{verdict_colour};
                          margin:.25rem 0;">{verdict_icon} {label}</div>
              <div style="color:#344054;">{verdict_description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_col, verdict_col = st.columns(2)
        with metric_col:
            st.metric("Estimated fake probability", f"{probability:.2%}")
        with verdict_col:
            st.metric("Model assessment", label)
        st.progress(probability, text="Estimated fake probability")
        st.caption("This is a model estimate, not a guarantee that a posting is fraudulent or legitimate.")

        st.markdown("#### What influenced this assessment?")
        reasons = result.get("reasons", [])
        if not reasons:
            st.info("The model did not return additional explanations.")
        for reason in reasons:
            reason_type = reason.get("type", "info")
            reason_text = reason.get("text", "")
            if reason_type == "warning":
                st.warning(reason_text)
            elif reason_type == "ok":
                st.success(reason_text)
            else:
                st.info(reason_text)

        with st.expander("Review submitted job details"):
            st.json(input_data)

        st.markdown(
            """
            <div class="panel">
              <strong style="color:#122744;">Stay safe</strong>
              <p style="margin:.4rem 0 0 0;color:#52657D;">
                Never pay a recruitment or registration fee to obtain a job.
                Verify the employer through independent official channels before sharing sensitive information.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error("The prediction could not be completed. Please check the model files and application dependencies.")
        with st.expander("Technical details"):
            st.code(str(exc))

st.divider()
st.caption(
    "Academic project | Fake Job Detection | Machine Learning Essentials"
)

# Professional dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #142D4E !important;
    }

    p, label, .stMarkdown {
        color: #334155;
    }

    
div.stButton > button,
div.stFormSubmitButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #17365D !important;
    color: #FFFFFF !important;
    border: 1px solid #17365D !important;
    border-radius: 8px;
    min-height: 44px;
    font-weight: 600;
}

div.stButton > button p,
div.stFormSubmitButton > button p,
div[data-testid="stFormSubmitButton"] > button p {
    color: #FFFFFF !important;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #244F7D !important;
    color: #FFFFFF !important;
    border-color: #244F7D !important;
}


    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #244F7D;
        color: white;
        border: 1px solid #244F7D;
    }

    div[data-testid="stForm"] {
        background-color: white;
        padding: 1.5rem;
        border: 1px solid #DCE4EE;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DCE4EE;
        border-radius: 10px;
        padding: 1rem;
    }

    div[data-testid="stProgressBar"] {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    hr {
        border-color: #DCE4EE;
    }
</style>
""", unsafe_allow_html=True)

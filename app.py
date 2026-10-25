
import streamlit as st
from src.predict import predict_job

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔎",
    layout="wide",
)


st.markdown(
    "<p style='color:#64748B; font-size:0.85rem; "
    "font-weight:600; letter-spacing:2px;'>"
    "JOB SAFETY TOOLKIT</p>",
    unsafe_allow_html=True,
)

st.title("🔎 Fake Job Detection")

st.markdown(
    "### Make informed decisions before you apply"
)

st.markdown(
    "Assess job postings for potential fraud indicators using "
    "machine learning and rule-based explanations."
)

st.divider()


mode = st.radio(
    "Choose input method",
    ["Detailed Form", "Quick Paste"],
    horizontal=True,
)


samples = {
    "Suspicious job — registration fee": {
        "title": "Work From Home Data Entry",
        "company_name": "",
        "location": "Remote",
        "employment_type": "Full-time",
        "salary_range": "₹50,000 per week",
        "company_profile": "",
        "description": (
            "Earn high income from home. No experience required. "
            "Pay a registration fee to start immediately. "
            "Limited vacancies available."
        ),
        "requirements": "Basic typing skills",
        "benefits": "",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
    },
    "Standard job — software engineer": {
        "title": "Software Engineer",
        "company_name": "Example Technology Ltd.",
        "location": "Ahmedabad, India",
        "employment_type": "Full-time",
        "salary_range": "₹6–10 LPA",
        "company_profile": (
            "Example Technology Ltd. develops business software "
            "for enterprise customers."
        ),
        "description": (
            "We are seeking a Software Engineer to develop, test, "
            "and maintain applications. Responsibilities include "
            "writing maintainable code, reviewing code, fixing bugs, "
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
    },
}

def load_sample():
    sample = samples.get(st.session_state.sample_choice)

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

st.markdown("#### Try an example")

sample_choice = st.selectbox(
    "Load a sample job posting",
    [
        "None — enter my own job",
        "Suspicious job — registration fee",
        "Standard job — software engineer",
    ],
    key="sample_choice",
    on_change=load_sample,
)

input_data = {}


if mode == "Quick Paste":
    pasted_text = st.text_area(
        "Paste the complete job posting",
        placeholder=(
            "Paste the job title, company information, "
            "description, requirements, salary, and other details..."
        ),
        height=300,
    )

    submitted = st.button(
        "Check Job Posting",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        if not pasted_text.strip():
            st.error("Please paste a job posting first.")
        else:
            input_data = {"description": pasted_text}

else:
    st.subheader("Job Posting Details")

    with st.form("job_details_form"):
        col1, col2 = st.columns(2)

        with col1:
            job_title = st.text_input(
                "Job Title *",
                placeholder="e.g. Software Engineer",
                key="job_title",
            )
            company_name = st.text_input(
                "Company Name",
                placeholder="Enter company name",
                key="company_name",
            )
            location = st.text_input(
                "Job Location",
                placeholder="e.g. Ahmedabad, India",
                key="location",
            )
            employment_type = st.selectbox(
                "Employment Type",
                [
                    "Not specified",
                    "Full-time",
                    "Part-time",
                    "Contract",
                    "Temporary",
                    "Internship",
                ],
                key="employment_type",
            )
            salary = st.text_input(
                "Salary / Compensation",
                placeholder="e.g. ₹5–8 LPA",
                key="salary"
            )

        with col2:
            company_profile = st.text_area(
                "Company Profile",
                placeholder="Describe the company...",
                height=130,
                key='company_profile',
            )
            requirements = st.text_area(
                "Job Requirements",
                placeholder="Skills, qualifications and experience...",
                height=130,
                key='requirements',         
            )
            benefits = st.text_area(
                "Benefits",
                placeholder="Health insurance, leave, etc.",
                height=90,
                key='benefits',
            )

        description = st.text_area(
            "Job Description *",
            placeholder="Describe the responsibilities and duties...",
            height=200,
            key='description',
        )

        col3, col4, col5 = st.columns(3)
        with col3:
            is_remote = st.checkbox("Remote / Telecommuting",
                                    key="is_remote",
                                    )
        with col4:
            has_logo = st.checkbox("Company logo available",
                                   key="has_logo",
                                   )
        with col5:
            has_questions = st.checkbox("Screening questions included",
                                            key="has_questions",
                                            )

        submitted = st.form_submit_button(
            "Check Job Posting",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not job_title.strip() or not description.strip():
            st.error(
                "Please enter both the Job Title and Job Description."
            )
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
                "employment_type": (
                    "" if employment_type == "Not specified"
                    else employment_type
                ),
            }


if input_data:
    st.divider()
    st.subheader("Job Risk Assessment")
    st.caption("Assessment generated by the machine learning model")

    try:
        with st.spinner("Analysing the job posting..."):
            result = predict_job(input_data)

        label = result["label"]
        probability = float(result["fake_probability"])
        probability = max(0.0, min(1.0, probability))

        # Verdict banner
        if label == "Likely Fake":
            st.error(f"🚨  {label}")
            verdict_colour = "#B91C1C"
            verdict_description = (
                "Multiple indicators suggest this posting may be fraudulent."
            )
        elif label == "Suspicious":
            st.warning(f"⚠️  {label}")
            verdict_colour = "#B45309"
            verdict_description = (
                "Review the warning signs carefully before proceeding."
            )
        else:
            st.success(f"✅  {label}")
            verdict_colour = "#166534"
            verdict_description = (
                "The model found fewer fraud indicators, but independent "
                "verification is still important."
            )

        st.markdown(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #DCE4EE;
                border-left:6px solid {verdict_colour};
                border-radius:12px;
                padding:20px;
                margin:12px 0 20px 0;">
                <div style="color:#64748B;font-size:13px;">
                    MODEL ASSESSMENT
                </div>
                <div style="
                    color:{verdict_colour};
                    font-size:28px;
                    font-weight:700;
                    margin:6px 0;">
                    {label}
                </div>
                <div style="color:#334155;">
                    {verdict_description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Fake Probability",
                f"{probability:.2%}",
            )

        with col2:
            st.metric(
                "Assessment",
                label,
            )

        st.markdown("#### Probability indicator")
        st.progress(probability)

        st.caption(
            "This percentage is the model's estimated fake probability, "
            "not a guarantee that the job is fraudulent or legitimate."
        )

        st.markdown("#### Explanation and warning signs")

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

        st.info(
            "Safety reminder: Never pay a recruitment or registration fee "
            "to obtain a job. Verify the employer through independent "
            "official channels before sharing sensitive information."
        )

    except Exception as exc:
        st.error(
            "The prediction could not be completed. "
            "Please check the model files and application dependencies."
        )
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

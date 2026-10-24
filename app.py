
import streamlit as st
from src.predict import predict_job

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Fake Job Detection")
st.caption(
    "An ML-powered tool to identify potentially "
    "fraudulent job postings."
)

st.info(
    "Enter job details or paste a complete job posting. "
    "The model will assess the posting and explain its findings."
)

mode = st.radio(
    "Choose input method",
    ["Detailed Form", "Quick Paste"],
    horizontal=True,
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
            )
            company_name = st.text_input(
                "Company Name",
                placeholder="Enter company name",
            )
            location = st.text_input(
                "Job Location",
                placeholder="e.g. Ahmedabad, India",
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
            )
            salary = st.text_input(
                "Salary / Compensation",
                placeholder="e.g. ₹5–8 LPA",
            )

        with col2:
            company_profile = st.text_area(
                "Company Profile",
                placeholder="Describe the company...",
                height=130,
            )
            requirements = st.text_area(
                "Job Requirements",
                placeholder="Skills, qualifications and experience...",
                height=130,
            )
            benefits = st.text_area(
                "Benefits",
                placeholder="Health insurance, leave, etc.",
                height=90,
            )

        description = st.text_area(
            "Job Description *",
            placeholder="Describe the responsibilities and duties...",
            height=200,
        )

        col3, col4, col5 = st.columns(3)
        with col3:
            is_remote = st.checkbox("Remote / Telecommuting")
        with col4:
            has_logo = st.checkbox("Company logo available")
        with col5:
            has_questions = st.checkbox("Screening questions included")

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
    st.subheader("Prediction Results")

    try:
        with st.spinner("Analysing the job posting..."):
            result = predict_job(input_data)

        label = result["label"]
        probability = float(result["fake_probability"])
        probability = max(0.0, min(1.0, probability))

        if label == "Likely Fake":
            st.error(f"🚨 Verdict: {label}")
        elif label == "Suspicious":
            st.warning(f"⚠️ Verdict: {label}")
        else:
            st.success(f"✅ Verdict: {label}")

        st.metric(
            "Estimated Fake Probability",
            f"{probability:.2%}",
        )
        st.progress(probability)

        st.caption(
            "This is a model estimate, not proof that a job is fraudulent. "
            "Verify the employer independently before sharing personal "
            "information or paying any fees."
        )

        st.markdown("### Why did the model give this result?")

        reasons = result.get("reasons", [])
        if not reasons:
            st.info("No additional explanation was returned.")

        for reason in reasons:
            reason_type = reason.get("type", "info")
            reason_text = reason.get("text", "")

            if reason_type == "warning":
                st.warning(reason_text)
            elif reason_type == "ok":
                st.success(reason_text)
            else:
                st.info(reason_text)

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

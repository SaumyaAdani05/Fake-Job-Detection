
import streamlit as st

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Fake Job Detection")
st.caption(
    "An ML-powered tool to help identify potentially "
    "fraudulent job postings."
)

st.info(
    "Enter the details of a job posting below. "
    "The prediction engine will be connected after "
    "the machine learning model is ready."
)

st.subheader("Job Posting Details")

with st.form("job_details_form"):
    job_title = st.text_input(
        "Job Title *",
        placeholder="e.g. Software Engineer",
    )

    company_name = st.text_input(
        "Company Name",
        placeholder="Enter the company name",
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

    company_profile = st.text_area(
        "Company Profile",
        placeholder="Describe the company...",
        height=120,
    )

    description = st.text_area(
        "Job Description *",
        placeholder="Paste the job description here...",
        height=220,
    )

    requirements = st.text_area(
        "Job Requirements",
        placeholder="Enter skills, qualifications and experience...",
        height=120,
    )

    salary = st.text_input(
        "Salary / Compensation",
        placeholder="e.g. ₹5–8 LPA",
    )

    submitted = st.form_submit_button(
        "Check Job Posting",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not job_title.strip() or not description.strip():
        st.error(
            "Please enter both the Job Title "
            "and Job Description."
        )
    else:
        st.success("Job posting details captured successfully.")

        st.subheader("Prediction Results")

        st.warning(
            "Prediction engine not connected yet. "
            "This is a placeholder, not a real assessment."
        )

        with st.expander("Review submitted details"):
            st.write("**Job Title:**", job_title)
            st.write("**Company:**", company_name or "Not provided")
            st.write("**Location:**", location or "Not provided")
            st.write(
                "**Employment Type:**",
                employment_type,
            )
            st.write("**Salary:**", salary or "Not provided")
            st.write("**Company Profile:**")
            st.write(company_profile or "Not provided")
            st.write("**Job Description:**")
            st.write(description)
            st.write("**Requirements:**")
            st.write(requirements or "Not provided")

st.divider()

st.caption(
    "Academic project | Fake Job Detection | "
    "Machine Learning Essentials"
)

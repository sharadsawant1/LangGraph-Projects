"""Streamlit interface for the LangGraph resume shortlisting workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from graph import resume_graph

st.set_page_config(page_title="LangGraph Resume Shortlister", page_icon="🧭", layout="wide")

st.title("🧭 Resume Shortlister with LangGraph")
st.caption("PDF extraction → profile parsing → experience filter → skill scoring → ranking")
st.write(
    "Upload resumes, define the job's required skills, and review a ranked shortlist. "
    "The score is a transparent measure of required-skill coverage—not a hiring decision."
)

with st.sidebar:
    st.header("Job requirements")
    default_skills = "Python, SQL, Machine Learning, Pandas, Scikit-learn"
    required_skills_text = st.text_area(
        "Required skills (comma-separated)", value=default_skills, height=110,
        help="Use job-related skills that you can verify in the resume.",
    )
    min_experience = st.number_input(
        "Minimum experience (years)", min_value=0.0, max_value=60.0, value=0.0, step=0.5,
    )
    top_k = st.number_input("Candidates to show", min_value=1, max_value=500, value=5, step=1)
    st.divider()
    st.caption("Experience is read from explicitly stated years of experience. If absent, it is treated as 0 years.")

uploaded_files = st.file_uploader(
    "Upload resume PDFs", type=["pdf"], accept_multiple_files=True,
    help="Text-based PDFs are supported. Scanned/image-only PDFs require OCR before processing.",
)

process_clicked = st.button("Run LangGraph workflow", type="primary", use_container_width=False)

if process_clicked:
    skills = [skill.strip() for skill in required_skills_text.split(",") if skill.strip()]
    uploaded_payload = []
    for uploaded_file in uploaded_files or []:
        uploaded_payload.append({"name": uploaded_file.name, "content": uploaded_file.getvalue()})

    with st.spinner("Running the resume workflow..."):
        try:
            state = resume_graph.invoke({
                "uploaded_files": uploaded_payload,
                "required_skills": skills,
                "min_experience": float(min_experience),
                "top_k": int(top_k),
            })
            st.session_state["resume_workflow_state"] = state
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")
            st.stop()

state = st.session_state.get("resume_workflow_state")
if state:
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])
    for error in errors:
        st.error(error)
    for warning in warnings:
        st.warning(warning)

    if not errors:
        extracted_count = len(state.get("extracted_resumes", []))
        parsed_count = len(state.get("parsed_resumes", []))
        eligible_count = len(state.get("eligible_resumes", []))
        results = state.get("results", [])

        metric_cols = st.columns(4)
        metric_cols[0].metric("Uploaded", len(state.get("uploaded_files", [])))
        metric_cols[1].metric("Readable PDFs", extracted_count)
        metric_cols[2].metric("Meet experience filter", eligible_count)
        metric_cols[3].metric("Shortlisted", len(results))

        st.subheader("Shortlisted candidates")
        if results:
            display_columns = [
                "name", "email", "experience_years", "match_score", "matched_skills",
                "missing_skills", "source",
            ]
            df = pd.DataFrame(results)[display_columns].rename(columns={
                "name": "Name",
                "email": "Email",
                "experience_years": "Experience (years)",
                "match_score": "Skill match (/10)",
                "matched_skills": "Matched required skills",
                "missing_skills": "Missing required skills",
                "source": "Source PDF",
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download shortlist as CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="resume_shortlist.csv",
                mime="text/csv",
            )

            st.subheader("Candidate details")
            for index, candidate in enumerate(results, start=1):
                title = f"{index}. {candidate['name']} — {candidate['match_score']}/10 skill match"
                with st.expander(title):
                    left, right = st.columns(2)
                    with left:
                        st.write(f"**Email:** {candidate.get('email') or 'Not found'}")
                        st.write(f"**Phone:** {candidate.get('phone') or 'Not found'}")
                        st.write(f"**Location:** {candidate.get('location') or 'Not found'}")
                        st.write(f"**Experience:** {candidate.get('experience_years', 0)} years (explicitly stated)")
                        st.write(f"**Education:** {candidate.get('education') or 'Not found'}")
                    with right:
                        st.write(f"**All detected skills:** {candidate.get('skills') or 'Not found'}")
                        st.write(f"**Matched required skills:** {candidate.get('matched_skills') or 'None detected'}")
                        st.write(f"**Missing required skills:** {candidate.get('missing_skills') or 'None'}")
                        st.write(f"**Source:** {candidate.get('source')}")
                    st.write("**Resume text preview**")
                    st.write(candidate.get("summary", "No preview available."))
        else:
            st.info("No candidates met the selected experience criterion or no readable resumes were found.")

        st.caption(
            "Review the original resumes before making decisions. PDF text extraction and skill detection can be incomplete; "
            "the score checks listed skill overlap only and does not assess overall candidate quality."
        )

with st.expander("How the LangGraph workflow works"):
    st.markdown(
        """
        1. **Validate inputs** — checks for uploaded resumes and required skills.
        2. **Extract resume text** — reads selectable text from each PDF.
        3. **Parse profiles** — extracts contact details, education, stated experience, and skills.
        4. **Filter by experience** — applies the minimum experience entered in the sidebar.
        5. **Score and rank** — scores skill coverage as `matched required skills / all required skills × 10`.

        Candidates with the same skill score are ordered by stated experience. Only job-related criteria are used in the score.
        """
    )

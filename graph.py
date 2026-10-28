"""LangGraph workflow for extracting, filtering, and ranking PDF resumes.

The graph uses transparent, deterministic criteria:
- minimum years of experience supplied by the user
- overlap between required skills and skills found in the resume

Scores are decision-support signals only. Always review original resumes before
making employment decisions.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any, TypedDict

from pypdf import PdfReader
from langgraph.graph import END, START, StateGraph


class ResumeWorkflowState(TypedDict, total=False):
    uploaded_files: list[dict[str, Any]]
    required_skills: list[str]
    min_experience: float
    top_k: int
    errors: list[str]
    warnings: list[str]
    extracted_resumes: list[dict[str, Any]]
    parsed_resumes: list[dict[str, Any]]
    eligible_resumes: list[dict[str, Any]]
    results: list[dict[str, Any]]


# Used only when a clearly labelled skills section cannot be found.
KNOWN_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "SQL", "MySQL",
    "PostgreSQL", "MongoDB", "React", "Angular", "Vue", "Django", "Flask",
    "FastAPI", "Streamlit", "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
    "Scikit-learn", "TensorFlow", "Keras", "PyTorch", "OpenCV", "NLP",
    "Natural Language Processing", "Computer Vision", "Machine Learning", "Deep Learning",
    "Generative AI", "GenAI", "LLM", "LLMs", "LangChain", "LangGraph", "RAG",
    "FAISS", "Hugging Face", "Transformers", "Power BI", "Tableau", "Excel",
    "Power Query", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "Linux",
    "HTML", "CSS", "REST API", "REST APIs", "GitHub", "Data Analysis", "ETL",
    "Apache Spark", "Hadoop", "Snowflake", "Statistics", "Data Visualization",
]

SECTION_HEADINGS = {
    "summary", "professional summary", "profile", "career objective", "objective",
    "skills", "technical skills", "key skills", "core skills", "competencies",
    "technical proficiencies", "education", "academic background", "qualifications",
    "educational qualifications", "experience", "work experience", "professional experience",
    "employment history", "internship", "internships", "projects", "personal projects",
    "certifications", "certificates", "achievements", "awards", "languages", "interests",
    "hobbies", "publications", "contact", "contact details", "personal details",
    "positions of responsibility", "tools", "technical expertise", "references",
}


def _normalise(value: str) -> str:
    """Normalize skill labels and a few common abbreviations for matching."""
    value = value.strip().lower()
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    value = re.sub(r"\s+", " ", value)
    aliases = {
        "ml": "machine learning",
        "machine-learning": "machine learning",
        "natural language processing": "nlp",
        "large language models": "llm",
        "large language model": "llm",
        "llms": "llm",
        "gen ai": "generative ai",
        "genai": "generative ai",
        "scikit learn": "scikit-learn",
        "powerbi": "power bi",
    }
    return aliases.get(value, value)


def _extract_section(text: str, headings: list[str]) -> str:
    """Extract a resume section without mistaking ordinary content lines for headings."""
    targets = {re.sub(r"\s+", " ", heading.strip().lower()) for heading in headings}
    known_headers = SECTION_HEADINGS | targets
    lines = text.splitlines()
    values: list[str] = []
    collecting = False

    # Longest-first avoids matching e.g. "skills" before "technical skills".
    ordered_targets = sorted(targets, key=len, reverse=True)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if collecting:
                values.append("")
            continue
        lowered = re.sub(r"\s+", " ", stripped).lower()
        header_name = lowered.split(":", 1)[0].strip() if ":" in lowered else lowered
        is_known_header = header_name in known_headers and (
            ":" in lowered or lowered == header_name
        )

        if collecting and is_known_header:
            break

        if not collecting:
            matched_target = None
            inline_content = ""
            for heading in ordered_targets:
                if lowered == heading:
                    matched_target = heading
                    break
                if lowered.startswith(heading + ":"):
                    matched_target = heading
                    inline_content = stripped.split(":", 1)[1].strip()
                    break
            if matched_target is not None:
                collecting = True
                if inline_content:
                    values.append(inline_content)
                continue
        else:
            values.append(stripped)

    return "\n".join(values).strip()


def _extract_skills(text: str) -> list[str]:
    section = _extract_section(
        text,
        ["skills", "technical skills", "key skills", "core skills", "competencies", "technical proficiencies"],
    )
    found: list[str] = []
    if section:
        # Keep common separators while avoiding making every word a skill.
        pieces = re.split(r"[,;•|\n]|\s+\u2022\s+", section)
        found.extend(piece.strip(" .:-\t") for piece in pieces if piece.strip(" .:-\t"))

    # When the resume has no parseable skills heading, use a known-skill scan.
    # The scan is also a fallback when a heading exists but produced no items.
    if not found:
        lower_text = text.lower()
        found = [skill for skill in KNOWN_SKILLS if re.search(
            rf"(?<![\w+#]){re.escape(skill.lower())}(?![\w+#])", lower_text
        )]

    # Deduplicate while preserving the spelling from the resume/skill catalogue.
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        key = _normalise(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _extract_experience(text: str) -> float:
    """Extract explicitly stated years of experience; do not infer from age/dates."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience\s*(?:of|:)\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        r"total\s+experience\s*:?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return 0.0


def _parse_resume(text: str, source: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = lines[0] if lines else "Unknown"
    # Avoid using a generic resume heading as the candidate name.
    if name.lower() in {"resume", "curriculum vitae", "cv", "profile"} and len(lines) > 1:
        name = lines[1]

    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    phone_match = re.search(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)", text)
    location_match = re.search(
        r"(?im)^\s*(?:current\s+location|location|address)\s*:\s*(.+?)\s*$", text
    )
    education = _extract_section(
        text, ["education", "academic background", "qualifications", "educational qualifications"]
    )
    education_lines = [line.strip(" •\t") for line in education.splitlines() if line.strip(" •\t")]

    return {
        "name": name[:120],
        "email": email_match.group(0) if email_match else "",
        "phone": re.sub(r"\s+", " ", phone_match.group(1)).strip() if phone_match else "",
        "location": location_match.group(1).strip() if location_match else "",
        "education": education_lines,
        "experience_years": _extract_experience(text),
        "skills_list": _extract_skills(text),
        "raw_text": text,
        "source": source,
    }


def validate_inputs(state: ResumeWorkflowState) -> dict[str, Any]:
    errors: list[str] = []
    if not state.get("uploaded_files"):
        errors.append("Upload at least one PDF resume.")
    if not state.get("required_skills"):
        errors.append("Enter at least one required skill, separated by commas.")
    if float(state.get("min_experience", 0)) < 0:
        errors.append("Minimum experience cannot be negative.")
    if int(state.get("top_k", 1)) < 1:
        errors.append("Number of candidates must be at least 1.")
    return {"errors": errors, "warnings": []}


def _route_after_validation(state: ResumeWorkflowState) -> str:
    return "stop" if state.get("errors") else "continue"


def extract_resume_text(state: ResumeWorkflowState) -> dict[str, Any]:
    extracted: list[dict[str, Any]] = []
    warnings = list(state.get("warnings", []))
    for upload in state.get("uploaded_files", []):
        name = str(upload.get("name") or "resume.pdf")
        content = upload.get("content", b"")
        if not name.lower().endswith(".pdf"):
            warnings.append(f"Skipped {name}: file is not a PDF.")
            continue
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                warnings.append(f"No selectable text found in {name}. It may be a scanned PDF; use OCR first.")
                continue
            extracted.append({"source": name, "raw_text": text})
        except Exception as exc:  # continue processing other candidate files
            warnings.append(f"Could not read {name}: {exc}")
    if not extracted:
        warnings.append("No readable resume text was extracted from the uploaded files.")
    return {"extracted_resumes": extracted, "warnings": warnings}


def parse_resumes(state: ResumeWorkflowState) -> dict[str, Any]:
    parsed = [
        _parse_resume(item["raw_text"], item["source"])
        for item in state.get("extracted_resumes", [])
    ]
    return {"parsed_resumes": parsed}


def filter_by_experience(state: ResumeWorkflowState) -> dict[str, Any]:
    minimum = float(state.get("min_experience", 0))
    eligible = [
        resume for resume in state.get("parsed_resumes", [])
        if float(resume.get("experience_years", 0)) >= minimum
    ]
    return {"eligible_resumes": eligible}


def score_and_rank(state: ResumeWorkflowState) -> dict[str, Any]:
    required = state.get("required_skills", [])
    required_by_key = {_normalise(skill): skill for skill in required if skill.strip()}
    required_keys = list(required_by_key)
    scored: list[dict[str, Any]] = []

    for resume in state.get("eligible_resumes", []):
        candidate_keys = {_normalise(skill): skill for skill in resume.get("skills_list", [])}
        matched_keys = [key for key in required_keys if key in candidate_keys]
        missing_keys = [key for key in required_keys if key not in candidate_keys]
        score = round(10 * len(matched_keys) / len(required_keys), 1) if required_keys else 0.0
        # Candidate summary is a short evidence snippet from the original resume.
        summary = re.sub(r"\s+", " ", resume.get("raw_text", "")).strip()
        scored.append({
            "name": resume.get("name", "Unknown"),
            "email": resume.get("email", ""),
            "phone": resume.get("phone", ""),
            "location": resume.get("location", ""),
            "education": " | ".join(resume.get("education", [])),
            "experience_years": float(resume.get("experience_years", 0)),
            "skills": ", ".join(resume.get("skills_list", [])),
            "matched_skills": ", ".join(required_by_key[key] for key in matched_keys),
            "missing_skills": ", ".join(required_by_key[key] for key in missing_keys),
            "matched_skill_count": len(matched_keys),
            "required_skill_count": len(required_keys),
            "match_score": score,
            "summary": summary[:600] + ("..." if len(summary) > 600 else ""),
            "source": resume.get("source", ""),
        })

    scored.sort(
        key=lambda item: (item["match_score"], item["experience_years"], item["matched_skill_count"]),
        reverse=True,
    )
    top_k = max(1, int(state.get("top_k", 5)))
    return {"results": scored[:top_k]}


def build_resume_graph():
    """Build and compile the LangGraph state machine."""
    builder = StateGraph(ResumeWorkflowState)
    builder.add_node("validate_inputs", validate_inputs)
    builder.add_node("extract_resume_text", extract_resume_text)
    builder.add_node("parse_resumes", parse_resumes)
    builder.add_node("filter_by_experience", filter_by_experience)
    builder.add_node("score_and_rank", score_and_rank)

    builder.add_edge(START, "validate_inputs")
    builder.add_conditional_edges(
        "validate_inputs",
        _route_after_validation,
        {"continue": "extract_resume_text", "stop": END},
    )
    builder.add_edge("extract_resume_text", "parse_resumes")
    builder.add_edge("parse_resumes", "filter_by_experience")
    builder.add_edge("filter_by_experience", "score_and_rank")
    builder.add_edge("score_and_rank", END)
    return builder.compile()


resume_graph = build_resume_graph()

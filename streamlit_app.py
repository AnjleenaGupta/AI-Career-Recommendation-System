import streamlit as st
import pandas as pd
import pdfplumber
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Career Recommender",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# HEADER UI
# -----------------------------
st.title("🚀 AI Career Recommendation System")
st.markdown("### Upload your resume or enter skills to get AI-powered career insights")

# -----------------------------
# DATA
# -----------------------------
df = pd.read_csv("careers.csv")

# -----------------------------
SKILLS_DB = [
    "python","sql","machine learning","deep learning","tensorflow",
    "pytorch","pandas","numpy","statistics","excel","power bi",
    "tableau","html","css","javascript","react","nodejs","mongodb",
    "c++","java","dsa","oops","dbms","nlp","computer vision",
    "aws","docker","kubernetes","linux","devops","spark","hadoop"
]

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(file):
    text = ""
    if file:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text().lower()
    return text

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    text = text.lower()
    return [s for s in SKILLS_DB if s in text]

# -----------------------------
# ML MODEL
# -----------------------------
def get_recommendation(user_input):
    data = df["skills"].tolist() + [user_input]

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(data)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])[0]

    temp = df.copy()
    temp["score"] = similarity

    return temp.sort_values("score", ascending=False)

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    user_skills = st.text_area("Enter your skills")

with col2:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Generate Career Report"):

    resume_text = extract_text(uploaded_file)
    resume_skills = extract_skills(resume_text)

    combined_skills = user_skills + " " + " ".join(resume_skills)

    result = get_recommendation(combined_skills)

    best = result.iloc[0]["career"]
    top_score = result.iloc[0]["score"] * 100

    # -------------------------
    # ATS SCORE (SMART)
    # -------------------------
    ats_score = min(100, int(top_score + len(resume_skills)*2))

    st.markdown("---")
    st.subheader("🎯 ATS SCORE")
    st.progress(ats_score)
    st.success(f"{ats_score}/100")

    # -------------------------
    # BEST CAREER
    # -------------------------
    st.subheader("🏆 Best Career Match")
    st.info(best)

    # -------------------------
    # TOP MATCHES
    # -------------------------
    st.subheader("📊 Top Career Matches")

    for _, row in result.head(5).iterrows():
        st.write(f"### {row['career']}")
        st.progress(int(row["score"] * 100))
        st.write(f"Match: {round(row['score']*100,2)}%")

    # -------------------------
    # SKILLS
    # -------------------------
    st.subheader("🧠 Extracted Skills from Resume")
    st.write(resume_skills)

    # -------------------------
    # MISSING SKILLS
    # -------------------------
    required = set(result.iloc[0]["skills"].split())
    user_set = set(combined_skills.split())

    missing = list(required - user_set)

    st.subheader("⚠️ Missing Skills to Improve")
    st.write(missing)

    # -------------------------
    # DOWNLOAD REPORT
    # -------------------------
    report = f"""
    AI CAREER REPORT

    Best Career: {best}
    ATS Score: {ats_score}/100

    Extracted Skills: {resume_skills}

    Missing Skills: {missing}
    """

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="career_report.txt",
        mime="text/plain"
    )

    # -------------------------
    # FOOTER
    # -------------------------
    st.markdown("---")
    st.success("🚀 Project Ready for Resume & Placement Submission")
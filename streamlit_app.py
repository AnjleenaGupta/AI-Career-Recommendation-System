import streamlit as st
import pandas as pd
import pdfplumber
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Career Recommender", layout="centered")

st.title("🚀 AI Career Recommendation System")
st.markdown("Upload your resume or enter skills to get best career suggestions.")

# -----------------------------
# SKILL DATABASE
# -----------------------------
SKILLS_DB = [
    "python","sql","machine learning","deep learning","tensorflow",
    "pytorch","pandas","numpy","statistics","excel","power bi",
    "tableau","html","css","javascript","react","nodejs","mongodb",
    "c++","java","dsa","oops","dbms","nlp","computer vision",
    "aws","docker","kubernetes","linux","devops","spark","hadoop"
]

# -----------------------------
# CAREER INFO
# -----------------------------
CAREER_INFO = {
    "Data Scientist": "₹8–20 LPA",
    "ML Engineer": "₹8–18 LPA",
    "AI Engineer": "₹8–20 LPA",
    "Frontend Developer": "₹4–10 LPA",
    "Full Stack Developer": "₹6–15 LPA",
    "Data Analyst": "₹4–12 LPA"
}

# -----------------------------
# RESUME TEXT EXTRACTION
# -----------------------------
def extract_text(file):
    text = ""
    if file is not None:
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
    return [skill for skill in SKILLS_DB if skill in text]

# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------
def recommend(user_input, df):

    df = df.copy()

    data = df["skills"].tolist()
    data.append(user_input)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    df["score"] = similarity[0]

    return df.sort_values(by="score", ascending=False)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("careers.csv")

# -----------------------------
# INPUT SECTION
# -----------------------------
user_skills = st.text_input("Enter your skills (python, sql, machine learning etc.)")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# -----------------------------
# BUTTON ACTION
# -----------------------------
if st.button("🚀 Recommend Career"):

    # Extract resume text
    resume_text = extract_text(uploaded_file)

    # Extract skills from resume
    resume_skills = extract_skills(resume_text)

    # Combine skills
    combined_skills = user_skills + " " + " ".join(resume_skills)

    # Get recommendations
    result = recommend(combined_skills, df)

    # -------------------------
    # RESULTS
    # -------------------------
    st.subheader("🏆 Top Career Matches")

    for _, row in result.head(5).iterrows():

        st.write(f"### {row['career']}")
        st.progress(float(row['score']))
        st.write(f"Match: {round(row['score']*100, 2)}%")

    # Best career
    best = result.iloc[0]["career"]

    st.subheader("🎯 Best Career Match")
    st.success(best)

    # Salary
    st.subheader("💰 Salary Range")
    st.info(CAREER_INFO.get(best, "Not Available"))

    # Skills
    st.subheader("🧠 Extracted Skills from Resume")
    st.write(resume_skills)
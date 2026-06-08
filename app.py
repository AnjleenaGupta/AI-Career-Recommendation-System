import os
os.makedirs("static", exist_ok=True)
import pdfplumber
import pandas as pd

from flask import Flask, render_template, request, send_file, url_for

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt

app = Flask(__name__)

# -----------------------------
# RESUME TEXT EXTRACTION
# -----------------------------
def extract_resume_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted.lower()
    return text


# -----------------------------
# SKILL EXTRACTION
# -----------------------------
SKILLS_DB = [
    "python","sql","machine learning","deep learning","tensorflow",
    "pytorch","pandas","numpy","statistics","excel","power bi",
    "tableau","html","css","javascript","react","nodejs","mongodb",
    "c++","java","dsa","oops","dbms","nlp","computer vision",
    "aws","docker","kubernetes","linux","devops","spark","hadoop"
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS_DB if skill in text]


# -----------------------------
# DATA
# -----------------------------
df = pd.read_csv("careers.csv")


ROADMAPS = {
    "Data Scientist": ["Python","SQL","Statistics","ML","Pandas","Visualization","Projects"],
    "ML Engineer": ["Python","ML","DL","TensorFlow","MLOps","Deployment"],
    "AI Engineer": ["Python","ML","NLP","CV","DL","Projects"],
    "Frontend Developer": ["HTML","CSS","JS","React","APIs","Projects"],
    "Full Stack Developer": ["Frontend","Backend","Database","Projects"],
    "Data Analyst": ["Excel","SQL","Power BI","Tableau","Projects"]
}


CAREER_INFO = {
    "Frontend Developer": {
        "description": "Builds websites using HTML, CSS, JS and React.",
        "salary": "₹4–10 LPA"
    },
    "Full Stack Developer": {
        "description": "Works on frontend + backend systems.",
        "salary": "₹6–15 LPA"
    },
    "Data Scientist": {
        "description": "Analyzes data and builds ML models.",
        "salary": "₹8–20 LPA"
    },
    "ML Engineer": {
        "description": "Builds and deploys ML systems.",
        "salary": "₹8–18 LPA"
    },
    "AI Engineer": {
        "description": "Works on NLP, CV and AI systems.",
        "salary": "₹8–20 LPA"
    },
    "Data Analyst": {
        "description": "Extracts insights using BI tools.",
        "salary": "₹4–12 LPA"
    }
}


# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------
def recommend_career(user_skills):

    user_skills = user_skills.lower()

    skill_data = df["skills"].tolist()
    skill_data.append(user_skills)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(skill_data)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    scores = similarity[0]

    temp_df = df.copy()
    temp_df["score"] = scores

    recommendations = temp_df.sort_values(by="score", ascending=False)

    top = recommendations.iloc[0]

    required_skills = set(top["skills"].split())
    user_set = set(user_skills.split())

    missing_skills = list(required_skills - user_set)
    ats_score = calculate_ats_score(user_skills, top["skills"])
    return (
        recommendations.head(5),
        top["career"],
        missing_skills,
        ROADMAPS.get(top["career"], []),
        CAREER_INFO.get(top["career"], {}),
        ats_score
    )


# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(best_career, extracted_skills, missing_skills, recommendations):
    
    os.makedirs("static", exist_ok=True)

    file_path = os.path.join("static", "career_report.pdf")
    doc = SimpleDocTemplate(file_path, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI Career Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Best Career:</b> {best_career}", styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Skills:</b>", styles['Heading2']))
    content.append(Paragraph(", ".join(extracted_skills), styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Missing Skills:</b>", styles['Heading2']))
    content.append(Paragraph(", ".join(missing_skills), styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Top Careers:</b>", styles['Heading2']))

    for _, row in recommendations.iterrows():
        content.append(Paragraph(
            f"{row['career']} - {round(row['score']*100,2)}%",
            styles['Normal']
        ))

    doc.build(content)
    return file_path
# -----------------------------
# CHART GENERATOR
# -----------------------------
def generate_chart(recommendations):

    os.makedirs("static", exist_ok=True)

    careers = recommendations["career"].head(5)
    scores = recommendations["score"].head(5) * 100

    plt.figure(figsize=(7,4))
    plt.bar(careers, scores)

    plt.title("Top Career Matches")
    plt.xticks(rotation=25)

    path = os.path.join("static", "chart.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


# -----------------------------
# MAIN ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = None
    best_career = None
    missing_skills = []
    extracted_skills = []
    career_roadmap = []
    career_details = {}

    if request.method == "POST":

        user_skills = request.form.get("skills", "")
        resume = request.files.get("resume")

        resume_text = ""

        if resume and resume.filename != "":
            resume_text = extract_resume_text(resume)

        resume_skills = extract_skills(resume_text)

        extracted_skills = resume_skills

        combined_skills = user_skills + " " + " ".join(resume_skills)

        (
            recommendations,
            best_career,
            missing_skills,
            career_roadmap,
            career_details,
            ats_score
        ) = recommend_career(combined_skills)
        print("PDF GENERATION STARTED")
        # PDF
        generate_pdf(
            best_career,
            extracted_skills,
            missing_skills,
            recommendations)

        # CHART
        generate_chart(recommendations)

    return render_template(
        "index.html",
        recommendations=recommendations,
        best_career=best_career,
        missing_skills=missing_skills,
        extracted_skills=extracted_skills,
        career_roadmap=career_roadmap,
        career_details=career_details,
        ats_score=ats_score
    )
def calculate_ats_score(user_skills, top_skills):
    
    user_set = set(user_skills.lower().split())
    career_set = set(top_skills.lower().split())

    if len(career_set) == 0:
        return 0

    match = len(user_set.intersection(career_set))
    score = (match / len(career_set)) * 100

    return round(score, 2)

# -----------------------------
# DOWNLOAD ROUTE
# -----------------------------
@app.route("/download-report")
def download_report():

    file_path = os.path.join("static", "career_report.pdf")

    if not os.path.exists(file_path):
        return "Report not generated yet. Please submit skills first."

    return send_file(file_path, as_attachment=True)
    


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)    
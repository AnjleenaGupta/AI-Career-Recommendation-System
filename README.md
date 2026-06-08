# 🚀 AI Career Recommendation System

An AI-powered web application that recommends the most suitable career paths based on user skills and uploaded resume using Machine Learning (TF-IDF + Cosine Similarity) and NLP.

---

## 📌 Features

- 🔍 Skill-based career recommendation using ML
- 📄 Resume upload (PDF parsing)
- 🧠 Automatic skill extraction from resume
- 📊 Top career match with percentage score
- 📉 Missing skills analysis
- 🛣️ Career learning roadmap
- 💰 Salary insights (India)
- 📑 Auto-generated PDF career report
- 📥 Downloadable report feature
- 📈 Progress bar UI for match visualization

---

## 🛠️ Tech Stack

- Python
- Flask
- Pandas
- Scikit-learn
- NLP (TF-IDF, Cosine Similarity)
- pdfplumber
- ReportLab
- HTML, CSS

---

## 🧠 How It Works

1. User enters skills or uploads resume
2. Resume text is extracted using `pdfplumber`
3. Skills are detected using keyword matching
4. ML model compares skills with dataset using TF-IDF
5. Cosine similarity calculates career match score
6. System returns:
   - Top careers
   - Best match
   - Missing skills
   - Roadmap
   - Salary info
7. PDF report is generated for download

---

## 📂 Project Structure

import os
import re
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Resume Ranking API")


# -------------------------------
# JD Text
# -------------------------------

# JD text 
jd_text = "An AI/ML Engineer for freshers typically involves building, testing, and deploying machine learning models using Python and frameworks like TensorFlow or PyTorch. Key responsibilities include data preprocessing, model training, and collaborating with teams to integrate AI solutions. Candidates need a degree in CS/AI, strong math skills, and a portfolio of projects.Key Responsibilities for Freshers:Model Development: Designing, building, and training machine learning models.Data Handling: Constructing, preprocessing, and cleaning data pipelines.Experimentation: Running tests to optimize model accuracy, latency, and scalability.Collaboration: Working with senior developers to integrate AI models into applications.Documentation: Documenting findings and maintaining AI/ML workflows.Required Skills & Qualifications:Programming Languages: Strong Python proficiency (essential), familiarity with R or Java is a plus.ML Frameworks: Knowledge of TensorFlow, PyTorch, or scikit-learn.Theoretical Knowledge: Deep understanding of statistics, probability, and algorithms.Data Skills: Experience with data analysis and visualization (Matplotlib, Seaborn).Cloud Platforms: Basic knowledge of AWS, Azure, or Google Cloud is preferred.Education: Bachelor's or Master's degree in Computer Science, AI, or related fields."

# -------------------------------
# Extract text
# -------------------------------
def extract_text(file):
    if file.endswith(".pdf"):
        text_data = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_data.append(text)
                except Exception as e:
                    print(f"⚠️ Skipping page due to error: {e}")
                    continue
        
        return " ".join(text_data)
    
    elif file.endswith(".docx"):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    
    return ""

# -------------------------------
# Clean text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text

# Similarity scoring --> using TF-IDF and cosine similarity ---> but it only check keyword similaritynot the context of the text.
# -------------------------------
# Skill Extraction
# -------------------------------
skills_list = [
    "python", "java", "c", "c++", "sql",
    "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "scikit-learn",
    "matplotlib", "seaborn", "pandas", "numpy",
    "aws", "azure", "google cloud",
    "data analysis", "data visualization"
]

def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return list(set(found_skills))

def skill_score(jd_text, resume_text):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    
    if not jd_skills:
        return 0
    
    matched = set(jd_skills) & set(resume_skills)
    return round((len(matched) / len(jd_skills)) * 100, 2)

# -------------------------------
# Project Detection
# -------------------------------
project_keywords = ["project", "developed", "built", "created", "implemented"]

def project_score(text):
    count = sum([text.count(word) for word in project_keywords])
    
    if count >= 5:
        return 100
    elif count >= 3:
        return 70
    elif count >= 1:
        return 40
    else:
        return 10

# -------------------------------
# Experience Scoring
# -------------------------------
def experience_score(text):
    # detect years (like 1 year, 2 years, etc.)
    matches = re.findall(r'(\d+)\s+year', text)
    
    if matches:
        years = max([int(x) for x in matches])
        return min(years * 20, 100)  # cap at 100
    else:
        return 20  # fresher baseline

# -------------------------------
# TF-IDF Score
# -------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_score(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    docs = [jd_text, resume_text]
    tfidf_matrix = vectorizer.fit_transform(docs)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(score[0][0] * 100, 2)


# Embeddings -->  Semantic Meaning (ADVANCED)👉 Understanding meaning like:
#AI = Artificial Intelligence
#NLP = Natural Language Processing
#DL = Deep Learning
#🔹 3. Experience Matching
#Years of experience
#Relevant domain
#🔹 4. Context Matching

#Example:

#“Built ML model using Python”

#vs

#“Studied ML theoretically”

#👉 Both have “ML” but different value

#🔹 5. Quality Signals
#Grammar
#Structure
#Projects
#Impact (accuracy %, deployment, etc.)

# -------------------------------
# Semantic Score
# -------------------------------

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks 
def semantic_score(resume_text):
    try:
        global model, jd_embedding

        if 'model' not in globals():
            print("Loading semantic model... (one-time)")
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            jd_embedding = model.encode(jd_text)

        from sentence_transformers import util
        
        chunks = chunk_text(resume_text, chunk_size=200)
        scores = []

        for chunk in chunks:
            emb = model.encode(chunk)
            score = util.cos_sim(jd_embedding, emb)
            scores.append(float(score))

        # take best matching chunk
        final_score = max(scores) if scores else 0
        
        return round(final_score * 100, 2)

    except Exception as e:
        print("⚠️ Semantic failed:", e)
        return 0

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>Resume Ranking Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }
            h2 { text-align: center; }
            .upload-section { text-align: center; margin-bottom: 30px; }
            .dashboard { max-width: 800px; margin: 0 auto; }
            .resume-card { background: #1c1c1c; color: white; padding: 15px 20px; margin-bottom: 10px;
                           display: flex; justify-content: space-between; align-items: center; border-radius: 5px; }
            .resume-card span { font-weight: bold; }
            .score { font-size: 1.1em; }
            .high { color: #4caf50; }   /* green */
            .medium { color: #ffeb3b; } /* yellow */
            .low { color: #f44336; }    /* red */
            button { padding: 5px 10px; border: none; border-radius: 3px; cursor: pointer; background: #2196f3; color: white; }
        </style>
    </head>
    <body>
        <h2>Resume Review Dashboard</h2>
        <div class="upload-section">
            <input type="file" id="zipfile" accept=".zip" />
            <button onclick="upload()">Upload & Rank</button>
        </div>
        <div class="dashboard" id="dashboard"></div>

        <script>
            function getScoreClass(score) {
                if(score > 60) return "high";      // green
                else if(score >= 40) return "medium"; // yellow
                else return "low";                  // red
            }

            async function upload() {
                const fileInput = document.getElementById('zipfile');
                if(fileInput.files.length === 0){
                    alert("Select a ZIP file");
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                const dashboard = document.getElementById("dashboard");
                dashboard.innerHTML = "<p>Processing...</p>";

                const res = await fetch("/upload-resumes-zip/", { method: "POST", body: formData });
                const data = await res.json();

                dashboard.innerHTML = "";
                data.results.forEach((r, idx) => {
                    const card = document.createElement("div");
                    card.className = "resume-card";

                    const name = document.createElement("span");
                    name.innerText = `#${idx+1} ${r.file}`;

                    const score = document.createElement("span");
                    score.innerText = r.final_score + "%";
                    score.className = "score " + getScoreClass(r.final_score);

                    const viewBtn = document.createElement("button");
                    viewBtn.innerText = "View Resume";
                    viewBtn.onclick = () => alert("You can implement resume preview here!");

                    card.appendChild(name);
                    card.appendChild(score);
                    card.appendChild(viewBtn);

                    dashboard.appendChild(card);
                });
            }
        </script>
    </body>
    </html>
    """
    
# -------------------------------
# API Endpoint
# -------------------------------
import zipfile
from fastapi.responses import HTMLResponse

@app.post("/upload-resumes-zip/")
async def upload_resumes_zip(file: UploadFile = File(...)):
    import tempfile
    import os, shutil

    results = []

    # Create temporary folder to extract ZIP
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, file.filename)
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Loop through extracted files
        for root, dirs, files in os.walk(tmpdir):
            for f_name in files:
                if f_name.lower().endswith((".pdf", ".docx")):
                    file_path = os.path.join(root, f_name)
                    text = extract_text(file_path)
                    cleaned_text = clean_text(text if text else "")

                    # Scoring
                    keyword = calculate_score(jd_text, cleaned_text)
                    sem = semantic_score(cleaned_text)
                    skills = skill_score(jd_text, cleaned_text)
                    projects = project_score(cleaned_text)
                    exp = experience_score(cleaned_text)
                    final_score = 0.25*keyword + 0.35*sem + 0.25*skills + 0.1*projects + 0.05*exp

                    results.append({
                        "file": f_name,
                        "final_score": round(final_score, 2),
                        "keyword_score": keyword,
                        "semantic_score": sem,
                        "skill_match": skills,
                        "project_score": projects,
                        "experience_score": exp
                    })

    # Sort results
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return JSONResponse(content={"results": results})

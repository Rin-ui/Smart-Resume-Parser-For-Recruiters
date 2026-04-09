from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import pdfplumber
from docx import Document
import os, shutil, tempfile, zipfile, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Resume Ranking API")

# -------------------------------
# Extract text
# -------------------------------

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text_data = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_data.append(text)
                except:
                    continue
        return " ".join(text_data)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
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

def extract_skills_from_jd(jd_text):
    jd_clean = clean_text(jd_text)
    possible_skills = [
        "python", "java", "c", "c++", "sql",
        "machine learning", "deep learning", "nlp",
        "tensorflow", "pytorch", "scikit-learn",
        "matplotlib", "seaborn", "pandas", "numpy",
        "aws", "azure", "google cloud",
        "data analysis", "data visualization"
    ]
    return [skill for skill in possible_skills if skill in jd_clean]

def extract_skills_from_resume(resume_text):
    resume_clean = clean_text(resume_text)
    possible_skills = [
        "python", "java", "c", "c++", "sql",
        "machine learning", "deep learning", "nlp",
        "tensorflow", "pytorch", "scikit-learn",
        "matplotlib", "seaborn", "pandas", "numpy",
        "aws", "azure", "google cloud",
        "data analysis", "data visualization"
    ]
    return [skill for skill in possible_skills if skill in resume_clean]

def skill_score(jd_text, resume_text):
    jd_skills = extract_skills_from_jd(jd_text)
    resume_skills = extract_skills_from_resume(resume_text)
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
    if count >= 5: return 100
    elif count >= 3: return 70
    elif count >= 1: return 40
    else: return 10

# -------------------------------
# Experience Scoring
# -------------------------------

def experience_score(text):
    matches = re.findall(r'(\d+)\s+year', text)
    if matches:
        years = max([int(x) for x in matches])
        return min(years*20, 100)
    else:
        return 20

# -------------------------------
# TF-IDF Score
# -------------------------------

def calculate_score(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(score[0][0]*100, 2)

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
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def semantic_score(jd_text, resume_text):
    try:
        global model
        from sentence_transformers import SentenceTransformer, util
        if 'model' not in globals():
            model = SentenceTransformer('all-MiniLM-L6-v2')
        jd_embedding = model.encode(jd_text)
        chunks = chunk_text(resume_text)
        scores = [float(util.cos_sim(jd_embedding, model.encode(c))) for c in chunks]
        return round(max(scores)*100 if scores else 0, 2)
    except:
        return 0

# -------------------------------
# API Endpoints + Frontend (handling dynamic JD Input and ZIP file upload)
# -------------------------------

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
            .score { font-size: 1.1em; }
            .high { color: #4caf50; }
            .medium { color: #ffeb3b; }
            .low { color: #f44336; }
            button { padding: 5px 10px; border: none; border-radius: 3px; cursor: pointer; background: #2196f3; color: white; }
        </style>
    </head>
    <body>
        <h2>Resume Review Dashboard</h2>
        <div class="upload-section">
            <textarea id="jdtext" placeholder="Paste JD here" rows="6" cols="80"></textarea><br><br>
            <input type="file" id="zipfile" accept=".zip" />
            <button onclick="upload()">Process Resumes</button>
        </div>
        <div class="dashboard" id="dashboard"></div>

        <script>
            function getScoreClass(score) {
                if(score > 60) return "high";
                else if(score >= 40) return "medium";
                else return "low";
            }

            async function upload() {
                const jdInput = document.getElementById('jdtext').value;
                const zipInput = document.getElementById('zipfile');

                if(!jdInput || zipInput.files.length===0){
                    alert("Provide JD text and ZIP file!");
                    return;
                }

                const formData = new FormData();
                formData.append("jd_text", jdInput);
                formData.append("zip_file", zipInput.files[0]);

                const dashboard = document.getElementById("dashboard");
                dashboard.innerHTML = "<p>Processing...</p>";

                const res = await fetch("/upload-resumes-zip/", { method: "POST", body: formData });
                const data = await res.json();

                dashboard.innerHTML = "";
                data.results.forEach((r, idx)=>{
                    const card = document.createElement("div");
                    card.className="resume-card";

                    const name = document.createElement("span");
                    name.innerText=`#${idx+1} ${r.file}`;

                    const score = document.createElement("span");
                    score.innerText=r.final_score+"%";
                    score.className="score "+getScoreClass(r.final_score);

                    const viewBtn=document.createElement("button");
                    viewBtn.innerText="View Resume";
                    viewBtn.onclick=()=>alert("Resume preview not implemented.");

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

@app.post("/upload-resumes-zip/")
async def upload_resumes_zip(jd_text: str = Form(...), zip_file: UploadFile = File(...)):
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, zip_file.filename)
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        jd_text_clean = clean_text(jd_text)

        for root, dirs, files in os.walk(tmpdir):
            for f_name in files:
                if f_name.lower().endswith((".pdf", ".docx")):
                    file_path = os.path.join(root, f_name)
                    text = extract_text(file_path)
                    cleaned_text = clean_text(text if text else "")
                    keyword = calculate_score(jd_text_clean, cleaned_text)
                    sem = semantic_score(jd_text_clean, cleaned_text)
                    skills = skill_score(jd_text_clean, cleaned_text)
                    projects = project_score(cleaned_text)
                    exp = experience_score(cleaned_text)
                    final_score = 0.25*keyword + 0.35*sem + 0.25*skills + 0.1*projects + 0.05*exp
                    results.append({
                        "file": f_name,
                        "final_score": round(final_score,2),
                        "keyword_score": keyword,
                        "semantic_score": sem,
                        "skill_match": extract_skills_from_resume(cleaned_text),
                        "project_score": projects,
                        "experience_score": exp
                    })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return JSONResponse({"results": results})

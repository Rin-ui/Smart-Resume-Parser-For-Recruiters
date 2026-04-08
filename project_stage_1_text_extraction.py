# step 1: Text extraction + hard coded or manually created resume folder with resume 1,2,3 to check working
# resumes/
# ├── resume1.txt
# ├── resume2.txt
# ├── resume3.txt

 # text extraction part : 
 import pdfplumber
from docx import Document    (pip install python-docx)

def extract_text(file):
    if file.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    elif file.endswith(".docx"):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])



# This is the function...... now we'll use the function to extract code

import pdfplumber
from docx import Document

def extract_text(file):
    if file.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    elif file.endswith(".docx"):
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])


import os

folder = "resumes"

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    
    print("\n----------------------------")
    print("Processing:", file)
    
    text = extract_text(path)
    
    print("Extracted Text Preview:\n")
    print(text[:])  # to extract entire content 

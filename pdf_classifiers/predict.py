import os
import pdfplumber
import pandas as pd
from docx import Document
import joblib
import csv

from scipy.sparse import hstack, csr_matrix
from pdf2image import convert_from_path
import pytesseract

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

model = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

CSV_PATH = os.path.join(PROJECT_ROOT, "predictions.csv")

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    if not text.strip():
        images = convert_from_path(path)
        for img in images:
            text += pytesseract.image_to_string(img)
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)

def extract_text_from_excel(path):
    text = ""
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        text += df.astype(str).to_string() + "\n"
    return text

def extract_text(path):

    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)

    if path.lower().endswith(".docx"):
        return extract_text_from_docx(path)

    if path.lower().endswith((".xlsx", ".xls")):
        return extract_text_from_excel(path)
    return ""

def format_features(text):
    text = text.lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    total = max(len(lines),1)
    question_lines = sum(1 for l in lines if "?" in l)
    qa_lines = sum(
        1 for l in lines
        if l.startswith(("q:", "q.", "question"))
        or l.startswith(("a:", "ans", "answer"))
    )
    faq_keywords = [
        "faq",
        "frequently asked",
        "what",
        "how",
        "when",
        "where",
        "why",
        "can i",
        "do i",
        "is it",
        "are there"
    ]
    faq_count = sum(
        1 for l in lines if any(w in l for w in faq_keywords)
    )

    step_words = [
        "step","click","enter","login","select",
        "upload","download","navigate","open","submit","press","button"
    ]
    step_count = sum(
        1 for l in lines if any(w in l for w in step_words)
    )

    index_count = sum(1 for l in lines if "index" in l)
    numbered_sections = sum(
        1 for l in lines
        if l[:2].strip().isdigit() and "." in l[:4]
    )

    policy_keywords = [
        "policy","eligibility","effective","scope","shall",
        "confidentiality","procedure","guideline","compliance",
        "authority","regulation","standard","validation",
        "authenticate","signature","certificate","trust",
        "digital signature"
    ]
    policy_count = sum(
        1 for l in lines if any(w in l for w in policy_keywords)
    )

    guide_keywords = [
        "guide","help guide","manual","process","process flow",
        "workflow","checklist","instruction","procedure","steps",
        "how to","clearance""initiate","submit"
    ]
    guide_count = sum(
        1 for l in lines if any(w in l for w in guide_keywords)
    )

    return csr_matrix([[
        question_lines / total,
        qa_lines / total,
        faq_count / total,
        guide_count / total,
        policy_count / total
    ]])

def predict(file_path):
    text = extract_text(file_path)
    if not text.strip():
        return "Unknown"
    tfidf = vectorizer.transform([text])
    extra = format_features(text)
    X = hstack([tfidf, extra])
    pred = model.predict(X)[0]

    label_map = {   
        0:"FAQ",
        1:"Policy",
        2:"User_Guide"
    }
    return label_map[pred]

if __name__ == "__main__":

     while True:
        path = input("Enter file path (or exit): ").strip()
        if path.lower() == "exit":
            break
        if not os.path.exists(path):
            print("File not found\n")
            continue
        prediction = predict(path)
        filename = os.path.basename(path)
        print("Prediction:", prediction, "\n")

        with open(CSV_PATH, "a",newline="") as f:
            writer = csv.writer(f)  
            writer.writerow([filename,prediction])
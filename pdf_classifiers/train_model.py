import os
import pdfplumber
import pandas as pd
from docx import Document
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
from pdf2image import convert_from_path
import pytesseract

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(PROJECT_ROOT, "data")

FAQ_PATH = os.path.join(BASE_PATH, "faq")
POLICY_PATH = os.path.join(BASE_PATH, "policy")
USER_GUIDE_PATH = os.path.join(BASE_PATH, "user_guides")

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

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
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            text += df.astype(str).to_string() + "\n"
    except:
        pass
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
    total = max(len(lines), 1)

    # FAQ signals
    question_lines = sum(1 for l in lines if "?" in l)

    qa_lines = sum(
        1 for l in lines
        if l.startswith(("q:", "q.", "question"))
        or l.startswith(("a:", "ans", "answer"))
    )

    faq_keywords = [
        "faq","frequently asked","what","how","when",
        "where","why","can i","do i","is it","are there"
    ]
    faq_count = sum(
        1 for l in lines if any(w in l for w in faq_keywords)
    )

    # User guide signals
    guide_keywords = [
        "guide","help guide","manual","process","process flow",
        "workflow","checklist","instruction","procedure","steps",
        "how to","clearance","initiate","submit","Relieving Letter",
        "onboarding","flowchart","index"
    ]

    guide_count = sum(
        1 for l in lines if any(w in l for w in guide_keywords)
    )

    numbered_sections = sum(
        1 for l in lines
        if l[:2].strip().isdigit() and "." in l[:4]
    )

    # Policy signals
    policy_keywords = [
        "policy","eligibility","effective","scope","shall",
        "confidentiality","procedure","guideline","compliance",
        "authority","regulation","standard","validation",
        "authenticate","signature","certificate","trust"
    ]

    policy_count = sum(
        1 for l in lines if any(w in l for w in policy_keywords)
    )

    return [
        question_lines / total,
        qa_lines / total,
        faq_count / total,
        guide_count / total,
        policy_count / total
    ]

def load_data():

    texts = []
    labels = []
    files = []

    for f in os.listdir(FAQ_PATH):
        path = os.path.join(FAQ_PATH, f)
        text = extract_text(path)

        if text.strip():
            texts.append(text)
            labels.append(0)
            files.append(f)

    for f in os.listdir(POLICY_PATH):
        path = os.path.join(POLICY_PATH, f)
        text = extract_text(path)

        if text.strip():
            texts.append(text)
            labels.append(1)
            files.append(f)

    for f in os.listdir(USER_GUIDE_PATH):
        path = os.path.join(USER_GUIDE_PATH, f)
        text = extract_text(path)

        if text.strip():
            texts.append(text)
            labels.append(2)
            files.append(f)

    return texts, labels, files

def train():

    texts, labels, files = load_data()
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        texts,
        labels,
        files,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5),
        max_features=5000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_extra = csr_matrix([format_features(t) for t in X_train])
    X_test_extra  = csr_matrix([format_features(t) for t in X_test])

    X_train_vec = hstack([X_train_tfidf, X_train_extra])
    X_test_vec = hstack([X_test_tfidf, X_test_extra])

    model = LinearSVC(class_weight="balanced", max_iter=6000)

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("\nTest Results\n")
    print(classification_report(
        y_test,
        preds,
        target_names=["FAQ","Policy","User_Guide"]
    ))

    label_map = {0:"FAQ",1:"Policy",2:"User_Guide"}
    print("\nMisclassified Files\n")
    for i in range(len(preds)):
        if preds[i] != y_test[i]:
            print("File:", f_test[i])
            print("Actual:", label_map[y_test[i]])
            print("Predicted:", label_map[preds[i]])
            print("-"*50)

    joblib.dump(model, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    print("\nModel saved in /model directory")

if __name__ == "__main__":
    train()
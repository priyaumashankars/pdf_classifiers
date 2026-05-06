import os
import sys
import pdfplumber
import pandas as pd
from docx import Document

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(PROJECT_ROOT, "data")

FAQ_PATH = os.path.join(BASE_PATH, "faq")
POLICY_PATH = os.path.join(BASE_PATH, "policy")
USER_GUIDE_PATH = os.path.join(BASE_PATH, "user_guides")

for path in [FAQ_PATH, POLICY_PATH, USER_GUIDE_PATH]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing required folder: {path}")

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
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
    except Exception:
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
    lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
    total = max(len(lines), 1)

    question_lines = sum(1 for l in lines if "?" in l)
    qa_lines = sum(
        1 for l in lines
        if l.startswith(("q:", "q.", "question"))
        or l.startswith(("a:", "ans", "answer"))
    )

    step_words = ["step", "click", "enter", "open", "login", "select", "navigate"]
    strong_action_words = ["click", "enter", "select", "upload", "download"]

    step_count = sum(1 for l in lines if any(w in l for w in step_words))
    strong_action_count = sum(1 for l in lines if any(w in l for w in strong_action_words))

    instructional_numbers = sum(
        1 for l in lines
        if l[:2].isdigit() and any(w in l for w in strong_action_words)
    )

    policy_keywords = [
        "policy",
        "eligibility",
        "effective",
        "applies to",
        "scope",
        "shall",
        "per annum",
        "confidentiality",
        "version",
        "approved",
        "rule"
    ]

    policy_count = sum(1 for l in lines if any(w in l for w in policy_keywords))
    return [
        question_lines / total,
        qa_lines / total,
        step_count / total,
        strong_action_count / total,
        instructional_numbers / total,
        policy_count / total
    ]

def train_model():
    texts, labels = [], []

    # FAQ → label 0
    for f in os.listdir(FAQ_PATH):
        if f.lower().endswith((".pdf", ".docx")):
            text = extract_text(os.path.join(FAQ_PATH, f))
            if text.strip():
                texts.append(text)
                labels.append(0)

    # POLICY → label 1
    for f in os.listdir(POLICY_PATH):
        if f.lower().endswith((".pdf", ".docx")):
            text = extract_text(os.path.join(POLICY_PATH, f))
            if text.strip():
                texts.append(text)
                labels.append(1)

    # USER GUIDES → label 2
    for f in os.listdir(USER_GUIDE_PATH):
        if f.lower().endswith(".pdf"):
            text = extract_text(os.path.join(USER_GUIDE_PATH, f))
            if text.strip():
                texts.append(text)
                labels.append(2)

    if len(texts) < 5:
        raise ValueError(
            "Not enough training data.\n"
            "Add more files to faq, policy, and user_guides folders."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=10000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_extra = csr_matrix([format_features(t) for t in X_train])
    X_test_extra = csr_matrix([format_features(t) for t in X_test])

    X_train_vec = hstack([X_train_tfidf, X_train_extra])
    X_test_vec = hstack([X_test_tfidf, X_test_extra])

    model = LinearSVC(
        class_weight={0: 1.3, 1: 1.0, 2: 1.2},
        max_iter=6000
    )
    model.fit(X_train_vec, y_train)

    print(
        classification_report(
            y_test,
            model.predict(X_test_vec),
            target_names=["FAQ", "Policy", "User_Guide"],
            zero_division=0
        )
    )
    return model, vectorizer

# PREDICTION
def predict_file(file_path, model, vectorizer):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith((".xlsx", ".xls")):
        return "FAQ"

    text = extract_text(file_path)

    if not text.strip():
        return "Policy"

    features = csr_matrix([format_features(text)])
    vec = hstack([
        vectorizer.transform([text]),
        features
    ])

    pred = model.predict(vec)[0]

    label_map = {
        0: "FAQ",
        1: "Policy",
        2: "User_Guide"
    }
    return label_map[pred]

def main():
    print("Training model...")
    model, vectorizer = train_model()
    print("\nModel ready!\n")

    while True:
        file_path = input("Enter file path to classify (or type 'exit'): ").strip()

        if file_path.lower() == "exit":
            print("Exiting...")
            break

        if not os.path.exists(file_path):
            print("❌ File not found. Try again.\n")
            continue

        prediction = predict_file(file_path, model, vectorizer)
        print(f"\n📄 Prediction: {prediction}\n")
        
if __name__ == "__main__":
    main()
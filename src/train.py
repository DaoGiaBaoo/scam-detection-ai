import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import json

# ===== LOAD DATA =====
data_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
# strip stray whitespace (and BOM) from the header before using column names
df = pd.read_csv(data_path, sep="|", encoding="utf-8-sig")
df.rename(columns=lambda col: col.strip(), inplace=True)

# ===== CLEAN TEXT =====
df["text"] = df["text"].astype(str).str.lower()
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

X = df["text"]
y = df["label"]

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=999
)

# ===== PIPELINE =====
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        max_features=10000,
        min_df=2
    )),
    ("svm", LinearSVC())
])

pipeline.fit(X_train, y_train)

# ===== EVALUATE =====
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ===== CV =====
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(scores, scores.mean())

# ===== TEST =====
while True:
    text = input("\nNhập text test: ")
    pred = pipeline.predict([text])[0]
    score = pipeline.decision_function([text])[0]
    print("SCAM" if pred == 1 else "NORMAL")
    print(score)
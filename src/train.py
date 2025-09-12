import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text
from data_loader import load_dataset

def train_and_save():
    df = load_dataset()
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, "logreg_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print("Model + Vectorizer saved.")

if __name__ == "__main__":
    train_and_save()

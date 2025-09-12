import joblib
from preprocess import clean_text

def predict_texts(texts):
    model = joblib.load("logreg_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")

    cleaned = [clean_text(t) for t in texts]
    X_tfidf = vectorizer.transform(cleaned)
    preds = model.predict(X_tfidf)

    return ["Real" if p==1 else "Fake" for p in preds]

if __name__ == "__main__":
    samples = [
        "Scientists discover cure for cancer!",
        "Click here to win $1000 now!"
    ]
    results = predict_texts(samples)
    for s, r in zip(samples, results):
        print(f"Text: {s}\nPrediction: {r}\n")

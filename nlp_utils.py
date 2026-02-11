import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_text_classifier(texts, labels):

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=500)
    model.fit(X, labels)
    
    joblib.dump(vectorizer, "models1/vectorizer.pkl")
    joblib.dump(model, "models1/text_classifier.pkl")
    print("Training complete and models saved.")

def predict_document_type(text):

    vectorizer = joblib.load("models1/vectorizer.pkl")
    model = joblib.load("models1/text_classifier.pkl")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    confidence = model.predict_proba(X).max()
    return pred, float(confidence)

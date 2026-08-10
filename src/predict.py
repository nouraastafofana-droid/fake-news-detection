from pathlib import Path
import joblib

from src.preprocessing import clean_text


MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "linear_svm_tfidf.joblib"
)

model = joblib.load(MODEL_PATH)


def predict_news(text):
    cleaned_text = clean_text(text)

    prediction = model.predict([cleaned_text])[0]
    decision_score = model.decision_function([cleaned_text])[0]

    label = "Real" if prediction == 1 else "Fake"

    return {"prediction": label, "decision_score": float(decision_score)}


if __name__ == "__main__":
    text = input("Enter a news article: ")

    result = predict_news(text)

    print("\nPrediction:", result["prediction"])
    print("Decision score:", result["decision_score"])

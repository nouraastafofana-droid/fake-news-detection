import streamlit as st

from src.predict import predict_news

st.set_page_config(
    page_title="Fake News Classifier", page_icon="📰", layout="centered"
)

st.title("Fake News Classifier")

st.write(
    "Enter a news article below to see how the trained model classifies it."
)

st.info(
    "This model was trained on the ISOT Fake News Dataset. "
    "Its prediction reflects patterns learned from this dataset and "
    "does not constitute factual verification or professional fact-checking."
)

article = st.text_area(
    "News article", placeholder="Paste a news article here...", height=250
)

if st.button("Classify article"):
    if article.strip():
        result = predict_news(article)

        prediction = result["prediction"]
        score = result["decision_score"]

        st.subheader("Prediction")

        if prediction == "Fake":
            st.error("FAKE")
        else:
            st.success("REAL")

        st.subheader("Decision score")
        st.metric("SVM decision score", f"{score:.3f}")

        st.caption(
            "Negative scores indicate the Fake side of the decision boundary, "
            "while positive scores indicate the Real side. "
            "The farther the score is from 0, the farther the prediction lies "
            "from the model's decision boundary. This score is not a probability."
        )
    else:
        st.warning("Please enter a news article.")

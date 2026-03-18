import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check if it is Fake or Real.")

# 🔥 Cache model so it doesn't retrain every time
@st.cache_resource
def train_model():
    fake = pd.read_csv("Fake.csv", encoding="utf-8", engine="python", on_bad_lines="skip")
    true = pd.read_csv("True.csv", encoding="utf-8", engine="python", on_bad_lines="skip")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true])
    df = df[["text", "label"]]
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1,2)
    )

    X_train = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    return model, vectorizer

model, vectorizer = train_model()

news = st.text_area("Paste News Text Here")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)
        probability = model.predict_proba(news_vector)

        confidence = max(probability[0])

        if prediction[0] == 0:
            st.error("🚨 Prediction: Fake News")
        else:
            st.success("✅ Prediction: Real News")

        st.write(f"Confidence Score: {confidence:.2f}")


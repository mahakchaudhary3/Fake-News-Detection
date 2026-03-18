import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# LOAD DATASET (ONLINE)
# -------------------------

@st.cache_data
def load_data():
    fake = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/Fake.csv")
    true = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true])
    df = df[["text", "label"]]
    df = df.dropna()

    return df


# -------------------------
# TRAIN MODEL
# -------------------------

@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, vectorizer, accuracy


# -------------------------
# MAIN APP
# -------------------------

st.title("📰 Fake News Detection System")

df = load_data()
model, vectorizer, accuracy = train_model(df)

st.write(f"Model Accuracy: {accuracy:.2f}")

st.write("Enter a news article below to check if it is Fake or Real.")

news = st.text_area("Paste News Text Here")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text")
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

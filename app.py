import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📊 WhatsApp Chat Analyzer with Emotion Detection")

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_saved():
    model = load_model("emotion_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_saved()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=["txt"])

def parse_chat(data):
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}.*?) - (.*?): (.*)"
    messages = []

    for line in data.split("\n"):
        match = re.match(pattern, line)
        if match:
            date, time, user, message = match.groups()
            messages.append([date, time, user, message])

    df = pd.DataFrame(messages, columns=["Date","Time","User","Message"])
    return df

# ---------------- CLEANING ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- PREDICTION ----------------
MAX_LEN = 50

def predict_emotion(texts):
    cleaned = [clean_text(t) for t in texts]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    preds = model.predict(padded, verbose=0)
    labels = label_encoder.inverse_transform(np.argmax(preds, axis=1))

    return labels

# ---------------- MAIN APP ----------------
if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    df = parse_chat(raw_text)

    st.subheader("Chat Preview")
    st.dataframe(df.head())

    st.subheader("Total Messages")
    st.write(len(df))

    st.subheader("Most Active Users")
    st.bar_chart(df["User"].value_counts())

    st.subheader("Word Cloud")
    text = " ".join(df["Message"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Emotion Detection")
    df["Emotion"] = predict_emotion(df["Message"].astype(str))
    st.dataframe(df[["User","Message","Emotion"]].head(20))

    st.subheader("Emotion Distribution")
    st.bar_chart(df["Emotion"].value_counts())

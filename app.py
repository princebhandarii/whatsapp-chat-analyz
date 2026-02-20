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

# Load saved model
@st.cache_resource
def load_saved():
    model = load_model("emotion_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_saved()


# File upload
uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=["txt"])

if uploaded_file:

    chat = uploaded_file.read().decode("utf-8")

    pattern = r'\[(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s?[APMapm\.]+)\]\s([^:]+):\s(.*)'
    messages = re.findall(pattern, chat)

    df = pd.DataFrame(messages, columns=['Date', 'Time', 'User', 'Message'])

    st.subheader("📌 Basic Statistics")
    col1, col2 = st.columns(2)

    col1.metric("Total Messages", len(df))
    col2.metric("Total Users", df['User'].nunique())

    st.subheader("👤 Top Active Users")
    st.bar_chart(df['User'].value_counts())

    # Wordcloud
    st.subheader("☁️ Word Cloud")
    text = " ".join(df['Message'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

    # Emotion Prediction
    st.subheader("😊 Emotion Analysis")

    def predict_emotion(text):
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=40)
        pred = model.predict(padded, verbose=0)
        return label_encoder.inverse_transform([np.argmax(pred)])[0]

    df['Predicted_Emotion'] = df['Message'].astype(str).apply(predict_emotion)

    st.bar_chart(df['Predicted_Emotion'].value_counts())

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

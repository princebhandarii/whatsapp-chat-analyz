import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model("emotion_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_files()

# ---------------- PAGE UI ----------------
st.set_page_config(page_title="Chat Emotion Analyzer", page_icon="💬", layout="wide")

st.title("💬 Chat Emotion Analyzer")
st.write("Upload your WhatsApp / Chat .txt file and AI will analyze emotions!")

uploaded_file = st.file_uploader("Upload chat file (.txt)", type=["txt"])


# ---------------- FUNCTION: CLEAN CHAT ----------------
def extract_messages(chat_text):
    lines = chat_text.split("\n")
    messages = []

    for line in lines:
        # WhatsApp format: 12/12/23, 10:32 PM - Name: message
        match = re.search(r" - .*?: (.*)", line)
        if match:
            msg = match.group(1)
            if len(msg) > 2:
                messages.append(msg)

    return messages


# ---------------- PREDICT FUNCTION ----------------
def predict_emotions(messages):
    results = []

    for text in messages:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

        prediction = model.predict(padded, verbose=0)

        idx = np.argmax(prediction)
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(np.max(prediction))

        results.append((text, emotion, confidence))

    df = pd.DataFrame(results, columns=["Message", "Emotion", "Confidence"])
    return df


# ---------------- MAIN APP ----------------
if uploaded_file:

    chat_data = uploaded_file.read().decode("utf-8")
    messages = extract_messages(chat_data)

    st.success(f"Found {len(messages)} messages")

    if st.button("Analyze Chat"):

        with st.spinner("AI is analyzing emotions..."):
            df = predict_emotions(messages)

        st.subheader("📊 Emotion Distribution")

        col1, col2 = st.columns(2)

        # Pie Chart
        with col1:
            emotion_counts = df["Emotion"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
            ax1.axis('equal')
            st.pyplot(fig1)

        # Bar Chart
        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x="Emotion", order=emotion_counts.index)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        st.subheader("🔥 Most Emotional Messages")

        top_msgs = df.sort_values("Confidence", ascending=False).head(10)
        st.dataframe(top_msgs)

        st.subheader("📝 Full Chat Analysis")
        st.dataframe(df)

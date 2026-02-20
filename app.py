import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import gdown
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

st.title("📊 WhatsApp Chat Analyzer with Emotion Detection")

# ---------------- DOWNLOAD MODEL ONLY ----------------

MODEL_URL = "https://drive.google.com/uc?id=1ZiN0mS7pptf2ksVR25acWf5OTo5gEG2n"


def download_model():
    if not os.path.exists("emotion_model.keras"):
        st.info("Downloading ML model... please wait ⏳")
        gdown.download(MODEL_URL, "emotion_model.keras", quiet=False)

download_model()

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

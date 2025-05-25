import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 50

# Load the saved BiLSTM model and tokenizer
model = load_model("bilstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Title
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment.")

# Text input from user
tweet = st.text_area("Tweet Text", "")

# Predict button
if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # Preprocess the tweet
        sequence = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict
        pred = model.predict(padded)[0][0]
        sentiment = "Positive 😊" if pred >= 0.5 else "Negative 😞"

        # Display result
        st.markdown(f"**Predicted Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** `{pred:.2f}`")

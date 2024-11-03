
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

# Load the model
with open('sentiment_pipeline_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("Sentiment Analysis")

# User input
user_input = st.text_input("Enter text for sentiment analysis:", "")

# Predict and display result
if user_input:
    prediction = loaded_model.predict([user_input])
    st.write("Prediction:", "Positive" if prediction[0] == 1 else "Negative")

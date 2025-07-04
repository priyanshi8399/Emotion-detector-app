import streamlit as st
from transformers import pipeline

# Load the emotion classification model
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_classifier = load_model()

# Title
st.title("🎭 AI Emotion Detector")
st.write("Enter a message below, and I'll tell you the emotion behind it using AI!")

# Text input
user_input = st.text_input("Type your message here:")

# On button click
if st.button("Analyze Emotion"):
    if user_input:
        result = emotion_classifier(user_input)
        emotion = result[0]['label']

        # Emoji Dictionary
        emoji_dict = {
            "joy": "😊",
            "anger": "😠",
            "sadness": "😢",
            "fear": "😨",
            "surprise": "😲",
            "love": "❤️"
        }

        emoji = emoji_dict.get(emotion.lower(), "🤔")
        st.success(f"Predicted Emotion: {emotion} {emoji}")
    else:
        st.warning("Please enter a message to analyze.")

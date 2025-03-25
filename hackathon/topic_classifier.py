import streamlit as st
from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

def classify_topic(text, candidate_topics):
    """
    Classify the input text into one of the candidate topics
    """
    result = classifier(text, candidate_topics)
    return result["labels"][0], result["scores"][0]  # Return the most likely topic and its confidence score

# Define candidate topics (customize these based on your needs)
default_topics = [
    "Fraud Issues", "Data Issues", "Backend Engineering", "Frontend Engineering", "Calling Issues",
    "Messaging Issues", "IT", "Product Quality Issues", "Customer Issues"]

# Streamlit UI
st.title("Topic Classifier")
st.write("Enter text to identify what topic it's about")

# User input
user_text = st.text_area("Enter your text here:", height=150)

# Allow users to customize topics
custom_topics = st.text_input("Customize topics (comma-separated):", ", ".join(default_topics))
topics = [topic.strip() for topic in custom_topics.split(",")]

if st.button("Classify") and user_text:
    with st.spinner("Analyzing..."):
        topic, confidence = classify_topic(user_text, topics)
    
    st.success(f"Topic: {topic}")
    st.info(f"Confidence: {confidence:.2%}")
    
    # Optional: Show confidence for all topics
    if st.checkbox("Show details"):
        result = classifier(user_text, topics)
        for label, score in zip(result["labels"], result["scores"]):
            st.write(f"- {label}: {score:.2%}")
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import openai
import streamlit as st
import requests
import google.generativeai as genai

# Load dataset (example: Kaggle credit card fraud dataset)
def load_data():
    url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    data = pd.read_csv(url)
    return data

# Preprocess data
def preprocess_data(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]
    return X, y

# Train Isolation Forest model
def train_isolation_forest(X):
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)
    return model

# Evaluate model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.where(y_pred == -1, 1, 0)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy Score:", accuracy_score(y, y_pred))
    return y_pred

# Generate insights using OpenAI API with Gemini fallback
def generate_insights(prompt):
    try:
        # Try OpenAI first
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.warning("OpenAI API failed, falling back to Gemini API")
        try:
            # Fallback to Gemini
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating insights: Both APIs failed. Please check your API keys and try again. Error: {str(e)}"

# Streamlit UI
def main():
    st.title("Fraudulent Transaction Detector")
    st.write("This app detects fraudulent transactions using Isolation Forest and provides insights using AI (OpenAI with Gemini fallback).")

    # Check for API keys in secrets
    if 'OPENAI_API_KEY' not in st.secrets or 'GEMINI_API_KEY' not in st.secrets:
        st.error("Please set your OpenAI and Gemini API keys in the secrets.")
        st.info("You can set them in your .streamlit/secrets.toml file:")
        st.code("""
        OPENAI_API_KEY = "your-openai-api-key"
        GEMINI_API_KEY = "your-gemini-api-key"
        """)
        return

    # Load data
    data = load_data()
    st.write("Dataset Sample:")
    st.write(data.head())

    # Preprocess data
    X, y = preprocess_data(data)

    # Train model
    if st.button("Train Model"):
        model = train_isolation_forest(X)
        st.success("Isolation Forest model trained successfully!")
        y_pred = evaluate_model(model, X, y)
        
        # Display results
        st.write("### Results")
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
        st.write(results_df.head(10))

    # Generate insights using AI
    st.write("### Generate Insights")
    prompt = st.text_input("Enter a prompt for insights (e.g., 'Explain why this transaction is fraudulent'):")
    if st.button("Generate Insights"):
        if prompt:
            with st.spinner("Generating insights..."):
                insights = generate_insights(prompt)
                st.write("*Insights:*")
                st.write(insights)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()

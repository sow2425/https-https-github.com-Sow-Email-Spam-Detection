import streamlit as st
import pandas as pd
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stopwd = stopwords.words('english')


def clean_text(text):

    text = text.lower()  # Lowercasing the text
    text = re.sub('-', ' ', text.lower())   # Replacing `x-x` as `x x`
    text = re.sub(r'http\S+', '', text)  # Removing Links
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuations
    text = re.sub(r'\s+', ' ', text)  # Removing unnecessary spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Removing single characters

    words = nltk.tokenize.word_tokenize(
        text, language="english", preserve_line=True)
    # Removing the stop words
    text = " ".join([i for i in words if i not in stopwd and len(i) > 2])

    return text.strip()


def load_vectorizer():
    return pickle.load(open('vectorizer.pkl', 'rb'))


# Streamlit UI
st.title("Email Spam Detection App")
st.caption("This app detects whether an email is spam or not.")
st.divider()

# Accuracies of models
showbtn = st.checkbox("Show model Accuracies", value=True)
if showbtn:
    st.markdown("## Model Accuracies")
    st.write(pickle.load(open('scores.pkl', 'rb')))


# Load models
model_files = ["LogisticRegression.pkl", "RandomForestClassifier.pkl",
               "SVC.pkl", "ComplementNB.pkl"]
model_options = {model_file.split(
    '.')[0]: model_file for model_file in model_files}
selected_model = st.selectbox("Select Model", list(model_options.keys()))

with open(model_options[selected_model], 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict spam or not
def predict_spam(input_text):
    input_text = clean_text(input_text)
    vectorizer = load_vectorizer()
    input_text = vectorizer.transform([input_text])
    prediction = model.predict(input_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"


# Input text or file upload
input_type = st.radio("Select Input Type", ["Text", "CSV File"])

if input_type == "Text":
    user_input = st.text_area("Enter the email text:")
    if st.button("Predict"):
        result = predict_spam(user_input)
        st.markdown(f"## Prediction: {result}")

elif input_type == "CSV File":
    st.markdown(
        "Note: :blue[The CSV file should have a column named 'Email']", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload the csv file here", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        try:
            df["Email"].head()
        except KeyError:
            st.error("The CSV file does not have a column named 'Email'")
            st.stop()
        
        with st.spinner("Predicting..."):
            predictions = df["Email"].apply(predict_spam)
            df["Prediction"] = predictions
            st.write("Predictions:")
            st.write(df)


# About this project
st.divider()
st.markdown("## About This Project")
st.write(
    "This Streamlit app is designed for email spam detection using different machine learning models."
)
st.write(
    "It allows users to select a model, input email text or upload a CSV file with emails, and "
    "provides predictions on whether each email is spam or not."
)

st.caption("This project is made by : Sowmya and Team")
st.write(
    "For more details, check the GitHub repository: [https://https://github.com/Sai-Siv/Email-Spam-Detection]"
)

# spam_classifier.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st

# Download stopwords (run only once)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam_data.csv', encoding='latin1')
df = df.rename(columns={'v1': 'label', 'v2': 'text'})

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])

df['clean_text'] = df['text'].apply(preprocess_text)

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Predict function using the trained model and vectorizer in memory
def predict_message(msg):
    msg_clean = preprocess_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    return model.predict(msg_vec)[0]

# --- Streamlit GUI code below ---

st.title("Spam Classifier")

email_input = st.text_area("Enter your email content here:")

if st.button("Classify"):
    if not email_input.strip():
        st.warning("Please enter some email text!")
    else:
        prediction = predict_message(email_input)
        st.success(f"Email classified as: {prediction}")

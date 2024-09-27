from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences
import regex as re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Initialize Flask app
app = Flask(__name__)

# Stopwords and Stemmer
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Define preprocessing functions
text_cleaning = r"\b0\S*|\b[^A-Za-z0-9]+"

def preprocess_filter(text, stem=False):
    text = re.sub(text_cleaning, " ", str(text.lower()).strip())
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                stemmer = SnowballStemmer(language='english')
                token = stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)

def one_hot_encoded(text, vocab_size=5000):
    return one_hot(text, vocab_size)

def word_embedding(text, max_length=40):
    preprocessed_text = preprocess_filter(text)
    encoded_text = one_hot_encoded(preprocessed_text)
    return pad_sequences([encoded_text], maxlen=max_length, padding='pre')

# Load the trained model
model = load_model('fake_news_detection_model.h5')  # Replace with the path to your saved model

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    max_length = 40  # Maximum length of the sequence as per your model's training

    # Preprocess and predict
    padded_sequence = word_embedding(input_text, max_length)
    prediction = model.predict(padded_sequence)
    result = np.where(prediction > 0.4, 1, 0)[0][0]

    if result == 1:
        output = 'Yes, this news is fake.'
    else:
        output = 'No, this news is not fake.'

    return render_template('index.html', prediction_text=output)

# Start the server
if __name__ == "__main__":
    app.run(debug=True)

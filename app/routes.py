from flask import Blueprint, render_template, request, flash
from preprocessing.text_cleaner import TextPreprocessor
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json

main_routes = Blueprint('main', __name__)

# Load models at startup
lr_model = joblib.load('models/trained_models/lr_model.pkl')
lstm_model = load_model('models/trained_models/lstm_model.h5')

with open('preprocessing/slang_emoji_mapping.json') as f:
    tokenizer_config = json.load(f)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
# (Note: In production, you'd load a saved tokenizer)

@main_routes.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    model_used = None
    
    if request.method == 'POST':
        text = request.form['text']
        model_choice = request.form['model']
        
        cleaner = TextPreprocessor()
        cleaned_text = cleaner.clean_text(text)
        
        if model_choice == 'logistic':
            prediction = lr_model.predict([cleaned_text])[0]
            model_used = "Logistic Regression"
        else:
            seq = tokenizer.texts_to_sequences([cleaned_text])
            pad = pad_sequences(seq, maxlen=100)
            prediction = (lstm_model.predict(pad) > 0.5).astype(int)[0][0]
            model_used = "LSTM Neural Network"
    
    return render_template('index.html', prediction=prediction, model_used=model_used)

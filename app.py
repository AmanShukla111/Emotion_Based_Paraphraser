import warnings
warnings.filterwarnings("ignore")

from utils import find_target_emotion, CPU_Unpickler, predict
from transformers import pipeline
emotion_classifier = pipeline('text-classification', model='bhadresh-savani/bert-base-go-emotion')

import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
import pickle
import io

app = Flask(__name__)

with open('Pickle_T5_Model.pkl', 'rb') as f:
    model = CPU_Unpickler(f).load()
    model.to(torch.device('cpu'))

with open('Pickle_T5_Tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def prediction():
    input_text = request.form.get('Sentence')
    prediction_text = predict(input_text, model, tokenizer)
    return render_template('index.html', prediction_text=f'Generated Output is: {prediction_text}')

if __name__=='__main__':
    app.run(debug=True)
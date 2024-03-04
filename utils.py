import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
emotion_classifier = pipeline('text-classification', model='bhadresh-savani/bert-base-go-emotion')

import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
import pickle
import io

#Function to create the input text to feed into the fine tuned model.
def find_target_emotion(text):
        emotion_mapping = {
        'amusement': 'surprise',
        'excitement': 'curiosity',
        'pride': 'admiration',
        'optimism': 'desire',
        'gratitude': 'relief',
        'joy': 'surprise',
        'love': 'admiration',
        'admiration': 'approval',
        'anger': 'annoyance',
        'disgust': 'disapproval',
        'grief': 'disappointment',
        'fear': 'nervousness',
        'sadness': 'disappointment',
        'approval': 'realization',
        'caring': 'approval',
        'desire': 'admiration',
        'relief': 'realization',
        'annoyance': 'confusion',
        'disapproval': 'confusion',
        'nervousness': 'confusion',
        'disappointment': 'nervousness',
        'embarrassment': 'annoyance',
        'remorse': 'embarrassment',
        'curiosity': 'confusion',
        'confusion': 'curiosity',
        'surprise': 'realization',
        }

        # Find the dominant emotion
        dominant_emotion = emotion_classifier(text)[0]['label']


        # Determine target emotion based on dominant emotion
        target_emotion = emotion_mapping[dominant_emotion]

        # Construct the output string
        output_string = f"{dominant_emotion} to {target_emotion}: {text}"
        return output_string


#Function to generate the emotion based paraphrase of the input text.
def predict(input_text, model, tokenizer):
    sample_input = find_target_emotion(input_text)
    # Tokenize the input
    tokenized_input = tokenizer(sample_input, return_tensors="pt", max_length=650, truncation=True)

    # Move input tensors to the CPU
    tokenized_input = {key: value.cpu() for key, value in tokenized_input.items()}

    # Generate output
    with torch.no_grad():
        generated_output = model.generate(
            **tokenized_input,
            max_length=400,  # Set the desired maximum length
            num_beams=4,     # You can adjust the number of beams for diverse outputs
        )

    # Postprocess the Output
    decoded_output = tokenizer.batch_decode(generated_output, skip_special_tokens=True)[0]

    return decoded_output

#Class to load model and tokenizer.
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        

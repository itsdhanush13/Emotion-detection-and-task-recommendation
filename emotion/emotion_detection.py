from transformers import pipeline

# Load NLP emotion analysis model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def analyze_text_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label'], result['score']


import librosa
import numpy as np

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    return mfccs


from deepface import DeepFace

def analyze_facial_expression(image_path):
    result = DeepFace.analyze(image_path, actions=['emotion'])
    return result[0]['dominant_emotion']

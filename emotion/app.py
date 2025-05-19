import streamlit as st
import cv2
from deepface import DeepFace
from transformers import pipeline
from task_recommendation import recommend_task
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np

# Load text emotion analysis model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Load audio emotion model
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

st.title("AI-Powered Task Optimizer")

# Function to analyze text-based emotion
def analyze_text_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label'], result['score']

# Function to capture an image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    st.write("Adjusting camera settings, please wait...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            cap.release()
            return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to capture image")
        return None

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=50)
    return frame

# Function to analyze facial expression
def analyze_facial_expression(image):
    cv2.imwrite("captured_face.jpg", image)
    result = DeepFace.analyze("captured_face.jpg", actions=['emotion'])[0]
    return result['dominant_emotion']

# ✅ Updated Function to record audio using sounddevice
def record_audio(duration=5, sample_rate=16000):
    st.write(f"🎤 Recording for {duration} seconds... Speak now!")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        audio_file_path = "recorded_audio.wav"
        write(audio_file_path, sample_rate, audio_data)
        return audio_file_path
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

# ✅ Updated Function to analyze emotion from audio with confidence
def analyze_audio_emotion(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return "Error", 0.0

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    emotion_labels = model.config.id2label
    return emotion_labels[predicted_class], confidence

# UI Components
st.header("Emotion Detection")

text_input = st.text_area("Enter your thoughts:", key="text_input")
if st.button("Analyze Text Emotion"):
    if text_input.strip():
        emotion, confidence = analyze_text_emotion(text_input)
        st.write(f"🧠 Text Emotion: **{emotion}** (Confidence: **{confidence:.2f}**)")
        st.write(f"🎯 Recommended Task: {recommend_task(emotion)}")
    else:
        st.warning("Please enter some text to analyze.")

if st.button("Capture Image from Webcam"):
    image = capture_image()
    if image is not None:
        st.image(image, channels="BGR")
        detected_emotion = analyze_facial_expression(image)
        st.write(f"🖼️ Facial Emotion: **{detected_emotion}**")
        st.write(f"🎯 Recommended Task: {recommend_task(detected_emotion)}")

if st.button("Record Audio"):
    st.info("Tip: Speak with clear emotion — e.g., happy, sad, angry — using expressive tone.")
    audio_file = record_audio()
    if audio_file:
        st.audio(audio_file)

        # Show waveform
        y, _ = librosa.load(audio_file, sr=16000)
        plt.figure(figsize=(10, 2))
        plt.plot(y)
        plt.title("Waveform")
        st.pyplot(plt)

        detected_emotion, confidence = analyze_audio_emotion(audio_file)
        st.write(f"🎙️ Audio Emotion: **{detected_emotion}** (Confidence: **{confidence:.2f}**)")
        st.write(f"🎯 Recommended Task: {recommend_task(detected_emotion)}")
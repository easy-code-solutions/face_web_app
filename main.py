
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import csv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
EMBED_DIR = 'embeddings'
ATTENDANCE_FILE = 'attendance.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

interpreter = tf.lite.Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def load_known_embeddings():
    embeddings = []
    names = []
    for file in os.listdir(EMBED_DIR):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(EMBED_DIR, file))
            name = os.path.splitext(file)[0]
            embeddings.append(emb)
            names.append(name)
    return embeddings, names

def mark_attendance(name):
    now = datetime.now()
    last_logged_time = None
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time'])

    with open(ATTENDANCE_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reversed(list(reader)):
            if row['Name'] == name:
                last_logged_time = datetime.strptime(f"{row['Date']} {row['Time']}", "%Y-%m-%d %H:%M:%S")
                break

    if not last_logged_time or (datetime.now() - last_logged_time) > timedelta(hours=3):
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['POST'])
def enroll():
    name = request.form['name']
    file = request.files['image']
    if not name or not file:
        return "Missing name or file", 400

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)
    emb = get_embedding(img_path)
    np.save(os.path.join(EMBED_DIR, f"{name}.npy"), emb)
    return f"✅ Enrolled '{name}' successfully."

@app.route('/identify', methods=['POST'])
def identify():
    file = request.files['image']
    if not file:
        return "Missing image", 400

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)
    emb = get_embedding(img_path).reshape(1, -1)

    known_embeddings, known_names = load_known_embeddings()
    if not known_embeddings:
        return "❌ No enrolled students found."

    sims = cosine_similarity(emb, known_embeddings)[0]
    max_idx = np.argmax(sims)
    max_sim = sims[max_idx]

    if max_sim > 0.75:
        name = known_names[max_idx]
        logged = mark_attendance(name)
        return render_template('result.html', name=name, confidence=max_sim, logged=logged)
    else:
        return render_template('result.html', name="Not Found", confidence=max_sim, logged=False)

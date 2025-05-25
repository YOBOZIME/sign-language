import os
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import re

# Chargement du corpus
base_dir = r"C:/Users/SWIFT 3/Documents/sign language/sign-language"
with open(os.path.join(base_dir, "mots.txt"), "r", encoding="utf-8") as f:
    corpus = [line.strip().lower() for line in f if line.strip()]

import random

def get_suggestions(prefix, mots, max_suggestions=3):
    if not prefix or not mots:
        return []
    prefix = prefix.lower()
    suggestions = [mot for mot in mots if mot.startswith(prefix)]
    random.shuffle(suggestions)  # Mélange aléatoirement
    return suggestions[:max_suggestions]


# Modèle
class HandSignClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Chargement du modèle
with open(os.path.join(base_dir, "label_encoder.pickle"), "rb") as f:
    label_encoder = pickle.load(f)
print(f"Loaded {len(corpus)} words from corpus")

model = HandSignClassifier(len(label_encoder.classes_))
model.load_state_dict(torch.load(os.path.join(base_dir, "model_nn.pth")))
model.eval()

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Vidéo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables
gesture_stability_delay = 1.0
max_history = 10
vote_threshold = 0.8
space_gesture = "space"
clear_gesture = "C"
backspace_gesture = "backspace"
current_word = []
last_char = None
last_sentence = ""
stable_gesture_start_time = None
history = deque(maxlen=max_history)
suggestions = []

# Initialisation
last_prediction_time = 0
prediction_interval = 0.4
clean_sentence = ''
pred_label = ''

# Couleurs
COLOR_BG = (30, 30, 60)
COLOR_TEXT = (240, 240, 240)
COLOR_GREEN = (0, 255, 150)
COLOR_BLUE = (255, 200, 100)
COLOR_RED = (50, 50, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Boîte autour de la main
        x_, y_ = [lm.x for lm in hand_landmarks.landmark], [lm.y for lm in hand_landmarks.landmark]
        min_x, max_x = min(x_), max(x_)
        min_y, max_y = min(y_), max(y_)
        x1, y1 = int(min_x * W), int(min_y * H)
        x2, y2 = int(max_x * W), int(max_y * H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BLUE, 2)

        # Landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Traitement du geste
        width, height = max_x - min_x, max_y - min_y
        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append((lm.x - min_x) / width if width > 0 else 0.0)
            data_aux.append((lm.y - min_y) / height if height > 0 else 0.0)

        if time.time() - last_prediction_time >= prediction_interval:
            last_prediction_time = time.time()

            with torch.no_grad():
                input_tensor = torch.tensor([data_aux], dtype=torch.float32)
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]
                pred_index = np.argmax(probs)
                pred_label = label_encoder.inverse_transform([pred_index])[0]

                # Vote majoritaire
                history.append(pred_label)
                votes = {c: history.count(c) / len(history) for c in set(history)}
                predicted_char = max(votes.items(), key=lambda x: x[1])[0] if max(votes.values()) >= vote_threshold else None

                if predicted_char:
                    if predicted_char != last_char:
                        stable_gesture_start_time = current_time
                        last_char = predicted_char
                    elif time.time() - stable_gesture_start_time >= gesture_stability_delay:
                        if predicted_char == clear_gesture:
                            current_word = []
                        elif predicted_char == backspace_gesture and current_word:
                            current_word.pop()
                        elif not current_word or current_word[-1] != predicted_char:
                            char_to_add = " " if predicted_char == space_gesture else predicted_char
                            current_word.append(char_to_add)
                            stable_gesture_start_time = current_time

    # Phrase et suggestions
    sentence = ''.join(current_word).strip()
    clean_sentence = re.sub(r'[^a-zA-Z\'-]', '', sentence).lower()

    if clean_sentence != last_sentence:
        suggestions = get_suggestions(clean_sentence, corpus)
        last_sentence = clean_sentence

    # Interface avec effet fondu
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (620, 200), COLOR_BG, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Affichage de la phrase
    cv2.putText(frame, f"Sentence: {sentence}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)

    # Suggestions
    for i, sug in enumerate(suggestions):
        y_pos = 100 + i * 35
        cv2.putText(frame, f"> {sug}", (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)

    # Lettre détectée
    if pred_label:
        cv2.putText(frame, f"Current: {pred_label}", (400, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 3)

    # Affichage
    cv2.imshow("Hand Sign Recognition", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import torch
import torch.nn as nn
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Config
gesture_stability_delay = 1.0
max_history = 10
vote_threshold = 0.8
space_gesture = "space"
clear_gesture = "C"
backspace_gesture = "backspace"

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

# Chargement
with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

model = HandSignClassifier(len(label_encoder.classes_))
model.load_state_dict(torch.load("model_nn.pth"))
model.eval()

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

current_word = []
last_char = None
stable_gesture_start_time = None
history = deque(maxlen=max_history)

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
        x_, y_ = [], []

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        min_x, max_x = min(x_), max(x_)
        min_y, max_y = min(y_), max(y_)
        width, height = max_x - min_x, max_y - min_y

        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append((lm.x - min_x) / width)
            data_aux.append((lm.y - min_y) / height)

        if len(data_aux) == 42:
            with torch.no_grad():
                input_tensor = torch.tensor([data_aux], dtype=torch.float32)
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]
                pred_index = np.argmax(probs)
                pred_label = label_encoder.inverse_transform([pred_index])[0]

                history.append(pred_label)
                votes = {c: history.count(c) / len(history) for c in set(history)}
                best_vote = max(votes.items(), key=lambda x: x[1])

                if best_vote[1] >= vote_threshold:
                    predicted_char = best_vote[0]
                else:
                    predicted_char = None

                if predicted_char:
                    display_char = " " if predicted_char == space_gesture else predicted_char
                    x1 = int(min_x * W) - 10
                    y1 = int(min_y * H) - 10
                    x2 = int(max_x * W) + 10
                    y2 = int(max_y * H) + 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, display_char, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                    if predicted_char != last_char:
                        stable_gesture_start_time = current_time
                        last_char = predicted_char
                    else:
                        if time.time() - stable_gesture_start_time >= gesture_stability_delay:
                            if predicted_char == clear_gesture:
                                current_word = []
                            elif predicted_char == backspace_gesture:
                                if current_word:
                                    current_word.pop()
                            elif not current_word or current_word[-1] != predicted_char:
                                char_to_add = " " if predicted_char == space_gesture else predicted_char
                                current_word.append(char_to_add)
                                stable_gesture_start_time = current_time

    cv2.putText(frame, f"Sentence: {''.join(current_word)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.imshow('Hand Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
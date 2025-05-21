import os
import cv2
import string

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = list(string.ascii_uppercase) + ['space', 'backspace']
dataset_size = 100  # Nombre d’images à ajouter par classe

cap = cv2.VideoCapture(0)

for label in classes:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class "{label}"')

    while True:
        ret, frame = cap.read()
        display_label = label.upper()
        cv2.putText(frame, f'Get ready for "{display_label}". Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    existing_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    counter = len(existing_files)
    added = 0

    while added < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f'{counter}.jpg'
        filepath = os.path.join(class_dir, filename)

        cv2.putText(frame, f'Capturing {display_label}: {added + 1}/{dataset_size}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.imwrite(filepath, frame)

        counter += 1
        added += 1

        if cv2.waitKey(25) & 0xFF == ord('e'):
            break

cap.release()
cv2.destroyAllWindows()



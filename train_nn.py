import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

# Chargement des données
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

X = np.array(data_dict['data'], dtype=np.float32)
y = np.array(data_dict['labels'])

# Encodage
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Filtrage classes rares (<2)
counts = Counter(y_encoded)
valid_labels = [label for label, count in counts.items() if count >= 2]
filtered = [i for i, label in enumerate(y_encoded) if label in valid_labels]
X = X[filtered]
y_encoded = y_encoded[filtered]

# Split
if min(Counter(y_encoded).values()) < 2:
    print("⚠️ Stratification désactivée.")
    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True)
else:
    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

# Tensor conversion
x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)
x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)

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

model = HandSignClassifier(len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement
for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/150, Loss: {loss.item():.4f}")

# Évaluation
model.eval()
with torch.no_grad():
    y_pred = torch.argmax(model(x_test_tensor), dim=1)
    acc = accuracy_score(y_test_tensor, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

# Sauvegarde
torch.save(model.state_dict(), 'model_nn.pth')
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)
import json
import os
# Chemin vers ton fichier mots.txt
base_dir = r"C:/Users/SWIFT 3/Documents/sign language/sign-language"
input_path = os.path.join(base_dir, "mots.txt")

# Lire le contenu JSON-like et le convertir
with open(input_path, "r", encoding="utf-8") as f:
    mots = json.loads(f.read())  # transforme la liste en Python list

# Réécrire un mot par ligne
with open(input_path, "w", encoding="utf-8") as f:
    for mot in mots:
        f.write(mot.strip().lower() + "\n")

print("✅ Fichier mots.txt corrigé avec un mot par ligne.")

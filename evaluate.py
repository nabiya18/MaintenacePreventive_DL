import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
X = np.load("X_sequences.npy")
y = np.load("y_labels.npy")

# Charger le modèle entraîné
model = load_model("modele_lstm_trained.keras")

# Prédictions
predictions = model.predict(X)
predictions = (predictions > 0.45).astype(int)  # Convertir en 0 ou 1

# Évaluation du modèle
accuracy = accuracy_score(y, predictions)
conf_matrix = confusion_matrix(y, predictions)
class_report = classification_report(y, predictions)

# Affichage des résultats
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualisation de la matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Pas de panne", "En panne"], yticklabels=["Pas de panne", "En panne"])
plt.xlabel("Prediction")
plt.ylabel("Actuel")
plt.title("Confusion Matrix")
plt.show()

# Visualisation des performances par classe
report_dict = classification_report(y, predictions, output_dict=True)
classes = list(report_dict.keys())[:-3]
f1_scores = [report_dict[cls]['f1-score'] for cls in classes]

plt.figure(figsize=(8, 5))
sns.barplot(x=classes, y=f1_scores, palette="viridis")
plt.xlabel("Classes")
plt.ylabel("F1-Score")
plt.title("Performance par classe")
plt.ylim(0, 1)
plt.show()

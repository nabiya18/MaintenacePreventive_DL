import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Charger les données
X = np.load("X_sequences.npy")
y = np.load("y_labels.npy")


# Définition du modèle amélioré
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),  # Couche supplémentaire
    Dense(10, activation='relu'),  # Ajout d'une nouvelle couche Dense
    Dense(1, activation='sigmoid')  # Sortie binaire
])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Sauvegarde du modèle avant entraînement
model.save("modele_lstm.keras")


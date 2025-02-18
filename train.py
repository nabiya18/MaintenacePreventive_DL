import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Activer l'exécution dynamique
tf.config.run_functions_eagerly(True)

# Charger les données
X = np.load("X_sequences.npy")
y = np.load("y_labels.npy")

# Charger le modèle
model = load_model("modele_lstm.h5")

# Définir un callback pour l'arrêt anticipé
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle (nécessaire après le chargement)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping]
)

# Sauvegarde du modèle entraîné
model.save("modele_lstm_trained.keras")


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Charger les données
df = pd.read_csv("../data/dataset_pannes.csv", parse_dates=["date"])
df.sort_values(by=["equipement_id", "date"], inplace=True)

# Sélection des caractéristiques et normalisation
features = ["temperature", "vibration", "courant", "energy_consumption", "cpu_usage", "memory_usage"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Création de séquences temporelles
sequence_length = 20  # Longueur des séquences
X, y = [], []

def create_sequences(data, labels, seq_length):
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(labels[i + seq_length])

for eq_id in df["equipement_id"].unique():
    eq_data = df[df["equipement_id"] == eq_id]
    create_sequences(eq_data[features].values, eq_data["panne"].values, sequence_length)

X, y = np.array(X), np.array(y)

# Sauvegarde des données prétraitées
np.save("X_sequences.npy", X)
np.save("y_labels.npy", y)

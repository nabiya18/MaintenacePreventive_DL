import numpy as np
import tensorflow as tf
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Charger le modèle
@st.cache_resource()
def load_trained_model():
    return load_model("C:/Users/Administrateur.DESKTOP-SGTCC1/MaintennancePredictive_POP/src/modele_lstm_trained.keras")

model = load_trained_model()

# Interface Web
st.title("📊 Prédiction de Maintenance des Équipements")

# Upload de fichier CSV
uploaded_file = st.file_uploader("Chargez un fichier CSV contenant les données", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Nettoyage des données : conversion en types numériques
    data = data.apply(pd.to_numeric, errors='coerce')  # Convertit les valeurs non numériques en NaN
    data = data.fillna(0)  # Remplace les NaN par 0

    expected_features = 11  # Nombre de caractéristiques attendues mis à jour
    sequence_length = 20  # Longueur de la séquence temporelle

    if "Panne" in data.columns:
        X = np.array(data.drop(columns=["Panne"]))  # Supprimer la colonne cible si existante
        y = np.array(data["Panne"])
    else:
        X = np.array(data)
        y = None

    # Vérifier que X a le bon nombre de caractéristiques
    if X.shape[1] != expected_features:
        st.error(f"Erreur : Le modèle attend {expected_features} colonnes, mais le fichier en contient {X.shape[1]}.")
    else:
        # Reformater X pour correspondre à (samples, 20, 11)
        try:
            num_samples = X.shape[0] // sequence_length  # Ajustement des échantillons
            X_reshaped = X[:num_samples * sequence_length].reshape(num_samples, sequence_length, expected_features)
            
            # Prédictions
            predictions = model.predict(X_reshaped)
            predictions = (predictions > 0.5).astype(int)
            
            st.subheader("📌 Résultats du modèle")
            if y is not None and len(y) >= num_samples * sequence_length:
                y = y[:num_samples * sequence_length].reshape(num_samples, sequence_length)
                y = y[:, -1]  # Prendre la dernière valeur de chaque séquence comme étiquette
                predictions = predictions[:, -1]  # Prendre la dernière prédiction de chaque séquence
                
                accuracy = accuracy_score(y, predictions)
                st.write(f"**Accuracy:** {accuracy:.4f}")

                conf_matrix = confusion_matrix(y, predictions)
                class_report = classification_report(y, predictions, output_dict=True)

                # Matrice de confusion
                st.subheader("📊 Matrice de confusion")
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"], ax=ax)
                plt.xlabel("Prédictions")
                plt.ylabel("Réel")
                st.pyplot(fig)

                # Performance par classe
                st.subheader("📈 Performance par classe")
                classes = list(class_report.keys())[:-3]
                f1_scores = [class_report[cls]["f1-score"] for cls in classes]

                fig, ax = plt.subplots()
                sns.barplot(x=classes, y=f1_scores, palette="viridis", ax=ax)
                plt.xlabel("Classes")
                plt.ylabel("F1-Score")
                plt.ylim(0, 1)
                st.pyplot(fig)
            else:
                st.write("Données sans étiquettes, impossible de calculer l'accuracy.")
        except ValueError as e:
            st.error(f"Erreur de mise en forme des données : {e}")


# Maintenance Prédictive des Équipements du POP

Ce projet utilise des techniques de Deep Learning pour prédire les pannes des équipements de télécommunications dans un centre POP.

## Étapes du projet
1. **Prétraitement des données** : Création de séquences temporelles à partir des données des équipements.
2. **Modélisation** : Utilisation d'un réseau de neurones récurrent (RNN) avec LSTM pour prédire les pannes.
3. **Entraînement** : Entraînement du modèle avec les données et sauvegarde du modèle final.
4. **Évaluation** : Évaluation de la performance du modèle sur les données de test.

## Démarrer le projet

1. Clonez ce repository.
2. Installez les dépendances : `pip install -r requirements.txt`
3. Téléchargez le dataset en suivant le lien fourni.
4. Exécutez `python src/train.py` pour entraîner le modèle.
5. Évaluez le modèle en exécutant `python src/evaluate.py`.

## Fichiers importants
- `data/maintenance_predictive_POP.csv` : Dataset simulé.
- `src/model.py` : Architecture du modèle.
- `src/train.py` : Script d'entraînement.
- `src/evaluate.py` : Évaluation du modèle.



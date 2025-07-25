# Financial Time Series Prediction 📈

Ce projet vise à prédire les séries temporelles financières (par exemple les prix des actions) à l'aide de techniques de Machine Learning et Deep Learning.

---

## 📁 Structure du projet

```
financial_time_series_prediction/
│
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py           # Chargement des données
│   ├── data_preprocessing.py       # Nettoyage et préparation
│   ├── feature_engineering.py      # Création de nouvelles features
│   ├── models.py                   # Définition des modèles
│   ├── prediction.py               # Prédictions et inférence
│   └── visualization.py            # Graphiques et visualisations
│
├── data/
│   ├── raw/                        # Données brutes
│   └── processed/                  # Données traitées
│
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Fichiers/dossiers à ignorer
├── main.py                         # Script principal
└── README.md                       # Ce fichier
```

---

## ⚙️ Installation

1. Clone le dépôt :
    ```bash
    git clone https://github.com/FayssalSabri/TimeSeriesForecast.git

    cd financial_time_series_prediction

    ```

2. Installe les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Comment exécuter

- Utilise `main.py` pour exécuter un pipeline complet.

---

## 🎯 Objectifs

- Extraire et nettoyer des données financières
- Explorer et visualiser les tendances
- Créer des features pertinentes
- Construire des modèles prédictifs (ARIMA, LSTM, RandomForest, Prophet, etc.)
- Évaluer les performances

---

## ✅ TODO

- Finaliser l'ingestion des données
- Mettre en place le pipeline de preprocessing
- Implémenter les premiers modèles
- Ajouter des métriques d'évaluation

---

## 📝 Auteurs

- SABRI Fayssal

---

## 📜 Licence

Ce projet est sous licence MIT — voir le fichier LICENSE pour plus d'informations.


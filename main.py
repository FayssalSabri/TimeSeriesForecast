import pandas as pd
from src.data_ingestion import fetch_data
from src.data_preprocessing import handle_missing_values, normalize_data
from src.feature_engineering import create_technical_indicators, create_lag_features
from src.models import train_arima_model, build_lstm_model, train_random_forest
from src.prediction import make_arima_predictions, evaluate_model, create_sequences
from src.visualization import plot_time_series, plot_predictions
import numpy as np

if __name__ == "__main__":

   
    # --- 1. Ingestion des données ---
    ticker = "BTC/USDT"  # Exemple avec Bitcoin
    start_date = "2023-01-01"
    # end_date = "2024-01-01"
    timeframe='1d'
    print(f"Téléchargement des données pour {ticker}...")
    df = fetch_data(ticker, start_date, timeframe)
    print("Données téléchargées.")
    
    # --- 2. Prétraitement des données ---
    df_cleaned = handle_missing_values(df.copy())
    print("Valeurs manquantes gérées.")

    # --- 3. Ingénierie des features (pour modèles ML/DL) ---
    df_features = create_technical_indicators(df_cleaned.copy())
    df_final = create_lag_features(df_features.copy(), lags=[1, 2, 3, 7]) # Exemple de lags
    print("Features créées.")

    # --- Préparation des données pour la modélisation ---
    # Pour ARIMA, on utilise la série directement
    series_for_arima = df_cleaned['close']

    # Pour LSTM, on prépare les séquences et on normalise
    data_for_lstm = df_final['close'].values
    scaled_data, scaler = normalize_data(data_for_lstm)
    
    time_step = 60 # Nombre de jours passés pour prédire le futur
    X, y = create_sequences(scaled_data, time_step)
    y = y.reshape(-1, 1)

    # Redimensionner X pour LSTM [samples, time_steps, features]
    X_lstm = X.reshape(X.shape[0], X.shape[1], 1)

    # Division train/test (chronologique)
    train_size_arima = int(len(series_for_arima) * 0.8)
    train_data_arima, test_data_arima = series_for_arima[0:train_size_arima], series_for_arima[train_size_arima:]

    train_size_lstm = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[0:train_size_lstm,:], X_lstm[train_size_lstm:len(X_lstm),:]
    y_train_lstm, y_test_lstm = y[0:train_size_lstm], y[train_size_lstm:len(y),:]

    # --- 4. Modélisation et Prédiction (Exemple ARIMA) ---
    print("\nEntraînement du modèle ARIMA...")
    # Il est recommandé de trouver le meilleur ordre (p, d, q) via auto_arima ou grid search
    arima_order = (5, 1, 0) 
    arima_model_fit = train_arima_model(train_data_arima, arima_order)
    print(arima_model_fit.summary())

    start_index = len(train_data_arima)
    end_index = len(series_for_arima) - 1
    arima_predictions = make_arima_predictions(arima_model_fit, start_index, end_index)
    arima_predictions.index = test_data_arima.index # Assurez-vous que les index correspondent

    arima_metrics = evaluate_model(test_data_arima.values, arima_predictions.values)
    print(f"\nMétriques ARIMA: {arima_metrics}")

    # --- Visualisation ARIMA ---
    plot_time_series(df_cleaned, 'close', f"Historique des Prix de {ticker}", "Prix ($)")
    plot_predictions(test_data_arima, arima_predictions, f"Prédictions ARIMA vs Réel pour {ticker}")

    # --- 4. Modélisation et Prédiction (Exemple LSTM) ---
    print("\nEntraînement du modèle LSTM...")
    lstm_model = build_lstm_model(input_shape=(time_step, 1))
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=64, verbose=1) # Augmentez les epochs pour de meilleurs résultats
    
    lstm_train_predict = lstm_model.predict(X_train_lstm)
    lstm_test_predict = lstm_model.predict(X_test_lstm)

    # Inverse la normalisation pour obtenir les valeurs réelles
    lstm_train_predict = scaler.inverse_transform(lstm_train_predict)
    y_train_lstm_inv = scaler.inverse_transform(y_train_lstm.reshape(-1,1))
    lstm_test_predict = scaler.inverse_transform(lstm_test_predict)
    y_test_lstm_inv = scaler.inverse_transform(y_test_lstm.reshape(-1,1))

    # Évaluation LSTM
    lstm_metrics_train = evaluate_model(y_train_lstm_inv, lstm_train_predict)
    lstm_metrics_test = evaluate_model(y_test_lstm_inv, lstm_test_predict)
    print(f"\nMétriques LSTM (Train): {lstm_metrics_train}")
    print(f"Métriques LSTM (Test): {lstm_metrics_test}")

    # Visualisation LSTM
    # Ajuster les indices pour la visualisation
    train_plot_index = df_final.index[time_step:train_size_lstm + time_step]
    test_plot_index = df_final.index[train_size_lstm + time_step + 1 : train_size_lstm + time_step + 1 + len(y_test_lstm_inv)]

    lstm_train_df = pd.DataFrame(lstm_train_predict, index=train_plot_index, columns=['Predicted'])
    lstm_test_df = pd.DataFrame(lstm_test_predict, index=test_plot_index, columns=['Predicted'])
    
    y_train_df = pd.DataFrame(y_train_lstm_inv, index=train_plot_index, columns=['Actual'])
    y_test_df = pd.DataFrame(y_test_lstm_inv, index=test_plot_index, columns=['Actual'])

    plot_predictions(y_train_df['Actual'], lstm_train_df['Predicted'], "Prédictions LSTM (Train) vs Réel")
    plot_predictions(y_test_df['Actual'], lstm_test_df['Predicted'], "Prédictions LSTM (Test) vs Réel")



    # Création de la cible pour RF (fermeture du jour suivant)
    df_final['target'] = df_final['close'].shift(-1)
    df_final = df_final.dropna()  # Supprimer les NaN apparus à cause du shift

    # Séparer features et cible
    X_rf = df_final.drop(columns=['target'])
    y_rf = df_final['target']

    # Train/test chronologique
    train_size_rf = int(len(df_final) * 0.8)
    X_train_rf, X_test_rf = X_rf.iloc[:train_size_rf, :], X_rf.iloc[train_size_rf:, :]
    y_train_rf, y_test_rf = y_rf.iloc[:train_size_rf], y_rf.iloc[train_size_rf:]

    # Entraînement RF
    rf_model = train_random_forest(X_train_rf, y_train_rf)

    # Prédictions
    rf_train_preds = rf_model.predict(X_train_rf)
    rf_test_preds = rf_model.predict(X_test_rf)

    # Évaluation
    rf_train_metrics = evaluate_model(y_train_rf.values, rf_train_preds)
    rf_test_metrics = evaluate_model(y_test_rf.values, rf_test_preds)

    plot_predictions(y_train_rf, pd.Series(rf_train_preds, index=y_train_rf.index), "Prédictions RF (Train) vs Réel")
    plot_predictions(y_test_rf, pd.Series(rf_test_preds, index=y_test_rf.index), "Prédictions RF (Test) vs Réel")

    print(f"Métriques RF Train: {rf_train_metrics}")
    print(f"Métriques RF Test: {rf_test_metrics}")
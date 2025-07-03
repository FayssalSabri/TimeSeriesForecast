from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def make_arima_predictions(model_fit: ARIMA, start: int, end: int) -> pd.Series:
    """Génère des prédictions avec un modèle ARIMA."""
    return model_fit.predict(start=start, end=end)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Évalue la performance du modèle."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def create_sequences(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Crée des séquences pour les modèles LSTM."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
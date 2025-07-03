from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_arima_model(data: pd.Series, order: tuple) -> ARIMA:
    """Entraîne un modèle ARIMA."""
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def build_lstm_model(input_shape: tuple) -> Sequential:
    """Construit et compile un modèle LSTM."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Entraîne un modèle RandomForestRegressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
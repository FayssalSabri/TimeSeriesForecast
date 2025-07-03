from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Gère les valeurs manquantes par interpolation linéaire."""
    return df.interpolate(method='linear')

def normalize_data(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalise les données à l'aide de MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler
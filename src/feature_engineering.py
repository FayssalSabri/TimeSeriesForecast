import pandas as pd
def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des indicateurs techniques au DataFrame."""
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    # Ajoutez d'autres indicateurs comme RSI, MACD si vous le souhaitez
    return df.dropna()

def create_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Crée des features basées sur les valeurs décalées (lagged values)."""
    for lag in lags:
        df[f'close_Lag_{lag}'] = df['close'].shift(lag)
    return df.dropna()
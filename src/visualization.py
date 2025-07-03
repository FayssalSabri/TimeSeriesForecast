import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_time_series(df: pd.DataFrame, column: str, title: str, ylabel: str):
    """Trace une série temporelle."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_predictions(y_true: pd.Series, y_pred: pd.Series, title: str):
    """Compare les valeurs réelles aux prédictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Réel')
    plt.plot(y_pred.index, y_pred, label='Prédictions', linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.show()
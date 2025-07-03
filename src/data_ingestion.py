import yfinance as yf
import pandas as pd
import ccxt
import pandas as pd

def fetch_data(symbol: str, since: str, timeframe: str = '1d') -> pd.DataFrame:
        exchange = ccxt.binance()
        since_timestamp = exchange.parse8601(since + 'T00:00:00Z')
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_timestamp)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.to_csv(f'data/raw/{symbol.replace("/", "")}_binance.csv')
        return df

def load_local_data(filepath: str) -> pd.DataFrame:
    """Charge des données à partir d'un fichier CSV local."""
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)
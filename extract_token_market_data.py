import pandas as pd
import requests
from time import sleep
from datetime import datetime

# Charger la liste des tokens
tokens_df = pd.read_csv("tokens_manuel.csv")

def fetch_binance_ohlcv(symbol, interval="1h", limit=72):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol
        df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"❌ Erreur pour {symbol} : {e}")
        return pd.DataFrame()

# Lancer la collecte
all_data = []

for _, row in tokens_df.iterrows():
    print(f"📥 Récupération : {row['symbol']}")
    df_token = fetch_binance_ohlcv(row["symbol"])
    if not df_token.empty:
        all_data.append(df_token)
    sleep(1)  # éviter le rate limit

if all_data:
    df_final = pd.concat(all_data)
    df_final.to_csv("market_data_tokens.csv", index=False)
    print("✅ Données sauvegardées dans market_data_tokens.csv")
else:
    print("⚠️ Aucun token n’a retourné de données")

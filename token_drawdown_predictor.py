# token_drawdown_predictor.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ----------- STEP 1: Fetch last 30 listed tokens from Binance Announcements -----------
def fetch_last_30_tokens():
    url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
    }
    tokens = []
    for page in range(1, 6):
        payload = {"catalogId": "48", "pageSize": 20, "pageNo": page}
        try:
            r = requests.post(url, json=payload, headers=headers)
            data = r.json()
            if not data or not data.get("data") or not data["data"].get("articles"):
                print(f"⚠️ Aucune donnée exploitable à la page {page}")
                continue
            for article in data["data"]["articles"]:
                if "Will List" in article["title"]:
                    title = article["title"]
                    token = title.split("Will List")[-1].split("(")[0].strip().split()[0]
                    if token not in tokens:
                        tokens.append(token)
                if len(tokens) >= 30:
                    break
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la page {page} : {e}")
            continue
    return tokens[:30]


# ----------- STEP 2: Get market data for each token -----------
def get_market_data(symbol):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1h", "limit": 72}
    try:
        r = requests.get(url, params=params)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["symbol"] = symbol
        return df
    except:
        return pd.DataFrame()

# ----------- STEP 3: Feature Engineering -----------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    df = df.sort_values(by=["symbol", "timestamp"])
    dataset = []
    for symbol in df["symbol"].unique():
        token_df = df[df["symbol"] == symbol].copy().reset_index(drop=True)
        token_df["rsi"] = compute_rsi(token_df["close"])
        token_df["variation_1h"] = token_df["close"].pct_change(periods=1) * 100
        token_df["variation_6h"] = token_df["close"].pct_change(periods=6) * 100
        token_df["variation_24h"] = token_df["close"].pct_change(periods=24) * 100

        for i in range(len(token_df) - 24):
            row = token_df.loc[i]
            window = token_df.iloc[i:i+24]
            min_price = window["low"].min()
            drawdown = (min_price - row["open"]) / row["open"] * 100
            dataset.append({
                "symbol": symbol,
                "open": row["open"],
                "close": row["close"],
                "volume": row["volume"],
                "rsi": row["rsi"],
                "variation_1h": row["variation_1h"],
                "variation_6h": row["variation_6h"],
                "variation_24h": row["variation_24h"],
                "drawdown_24h": drawdown
            })
    return pd.DataFrame(dataset).dropna()

# ----------- STEP 4: Train XGBoost model -----------
def train_model(df):
    features = ["open", "close", "volume", "rsi", "variation_1h", "variation_6h", "variation_24h"]
    X = df[features]
    y = df["drawdown_24h"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}%")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    return model

# ----------- STEP 5: Predict for a given token -----------
def predict_for_token(model, token_symbol):
    df = get_market_data(token_symbol)
    if df.empty:
        print("❌ Aucune donnée trouvée pour ce token.")
        return
    df["rsi"] = compute_rsi(df["close"])
    df["variation_1h"] = df["close"].pct_change(periods=1) * 100
    df["variation_6h"] = df["close"].pct_change(periods=6) * 100
    df["variation_24h"] = df["close"].pct_change(periods=24) * 100
    df = df.dropna().reset_index(drop=True)
    latest = df.iloc[-1]
    features = ["open", "close", "volume", "rsi", "variation_1h", "variation_6h", "variation_24h"]
    X = latest[features].values.reshape(1, -1)
    prediction = model.predict(X)[0]
    print(f"🔮 Prédiction pour {token_symbol} → Drawdown probable sur 24h : {prediction:.2f}%")

# ================== EXECUTION COMPLETE ==================
if __name__ == "__main__":
    print("📡 Récupération des tokens listés récemment...")
    tokens = ["PROMPT", "BABY", "OM", "FHE", "STONE","PUMP","FUN","GUN","FORTH","PAXG","MLN","STO"]
    # all_data = pd.concat([get_market_data(f"{t}USDT") for t in tokens if get_market_data(f"{t}USDT").shape[0] > 0], ignore_index=True)

    dfs = [get_market_data(f"{t}USDT") for t in tokens if get_market_data(f"{t}USDT").shape[0] > 0]
    if not dfs:
        print("❌ Aucun token n'a renvoyé de données. Script arrêté.")
        exit()
    all_data = pd.concat(dfs, ignore_index=True)

    print("🔧 Préparation des données...")
    dataset = prepare_features(all_data)
    print("🧠 Entraînement du modèle...")
    model = train_model(dataset)
    print("🤖 Prédiction personnalisée...")
    predict_for_token(model, "B3")

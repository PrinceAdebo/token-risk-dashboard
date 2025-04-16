# TOKEN RISK DASHBOARD - STREAMLIT + TELEGRAM + BINANCE + XGBOOST
# Auteur: [python] - Machine Learning Crypto Specialist

import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import xgboost as xgb
import telepot
import datetime
import streamlit as st

# ---------------------- 📡 Récupération des tokens ---------------------- #
def fetch_latest_binance_tokens(limit=30):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url).json()
        tickers = sorted(response, key=lambda x: float(x['openTime']), reverse=True)
        return [t['symbol'] for t in tickers if 'USDT' in t['symbol']][:limit]
    except Exception as e:
        print("Erreur API Binance", e)
        return []

# ---------------------- 📊 Métadonnées simulées ---------------------- #
def fetch_token_metadata(symbol):
    return {
        'market_cap': np.random.randint(1e6, 1e9),
        'twitter_followers': np.random.randint(1000, 500000),
        'telegram_members': np.random.randint(1000, 300000)
    }

# ---------------------- 📈 Données de marché ---------------------- #
def fetch_token_market_data(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=48"
    klines = requests.get(url).json()
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ---------------------- 🧠 Feature Engineering ---------------------- #
def compute_features(df):
    df['rsi_1h'] = RSIIndicator(close=df['close'], window=1).rsi()
    df['rsi_6h'] = RSIIndicator(close=df['close'], window=6).rsi()
    df['rsi_24h'] = RSIIndicator(close=df['close'], window=24).rsi()
    price_drop = (df['close'].iloc[47] - df['close'].iloc[0]) / df['close'].iloc[0] < -0.2
    return {
        'volume_1h': df['volume'][:1].sum(),
        'volume_6h': df['volume'][:6].sum(),
        'volume_24h': df['volume'][:24].sum(),
        'rsi_1h': df['rsi_1h'].iloc[0],
        'rsi_6h': df['rsi_6h'].iloc[6] if len(df) > 6 else np.nan,
        'rsi_24h': df['rsi_24h'].iloc[24] if len(df) > 24 else np.nan,
        'target': int(price_drop)
    }

# ---------------------- 🧹 Dataset complet ---------------------- #
def prepare_dataset(tokens):
    data = []
    for symbol in tokens:
        try:
            df = fetch_token_market_data(symbol)
            meta = fetch_token_metadata(symbol)
            features = compute_features(df)
            features.update(meta)
            features['token'] = symbol
            data.append(features)
        except Exception as e:
            continue
    df = pd.DataFrame(data).dropna()
    df.to_csv("dataset_simule.csv", index=False)
    return df

# ---------------------- 🔄 Entraînement du modèle ---------------------- #
def train_model(df):
    X = df.drop(columns=['target', 'token'])
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler

# ---------------------- 🔍 Analyse manuelle ---------------------- #
def analyze_token(symbol, model, scaler):
    df = fetch_token_market_data(symbol)
    meta = fetch_token_metadata(symbol)
    features = compute_features(df)
    features.update(meta)
    input_df = pd.DataFrame([features])
    X = scaler.transform(input_df.drop(columns=['target']))
    proba = model.predict_proba(X)[0][1]
    log_prediction(symbol, proba)
    return input_df, proba

# ---------------------- 📁 Log prédictions ---------------------- #
def log_prediction(symbol, probability):
    with open("log_predictions.csv", "a") as f:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{now},{symbol},{probability:.4f}\n")

# ---------------------- 📊 Streamlit Dashboard ---------------------- #
def launch_dashboard():
    import os

        # 🔧 Création du fichier log s'il n'existe pas
    if not os.path.exists("log_predictions.csv"):
            with open("log_predictions.csv", "w") as f:
                f.write("")  # ou écrire un entête : "timestamp,symbol,probability\\n"
    if not os.path.exists("dataset_simule.csv"):
        df_empty = pd.DataFrame(columns=["volume_1h", "volume_6h", "volume_24h",
                                         "rsi_1h", "rsi_6h", "rsi_24h",
                                         "market_cap", "twitter_followers", "telegram_members",
                                         "token", "target"])
        df_empty.to_csv("dataset_simule.csv", index=False)


    st.set_page_config(page_title="Token Risk Monitor", layout="wide")
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Accès", ["Historique", "Analyse", "Alertes"])

    if page == "Historique":
        st.title("📉 Historique des prédictions")
        try:
            df = pd.read_csv("log_predictions.csv", names=["timestamp", "symbol", "probability"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            seuil = st.slider("Seuil", 0.0, 1.0, 0.7, 0.01)
            symbol = st.selectbox("Token", ["Tous"] + sorted(df['symbol'].unique()))
            if symbol != "Tous":
                df = df[df['symbol'] == symbol]
            df = df[df['probability'] >= seuil]
            st.dataframe(df)
            fig, ax = plt.subplots()
            for token in df['symbol'].unique():
                sub = df[df['symbol'] == token]
                ax.plot(sub['timestamp'], sub['probability'], label=token)
            ax.axhline(0.7, color='red', linestyle='--', label='Seuil 0.7')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error("Erreur : " + str(e))

    elif page == "Analyse":
        st.title("🔍 Analyse manuelle d’un token")
        token_input = st.text_input("Symbole (ex: ARBUSDT)")
        if st.button("Analyser maintenant") and token_input:
            dataset = pd.read_csv("dataset_simule.csv")
            model, scaler = train_model(dataset)
            try:
                input_df, proba = analyze_token(token_input, model, scaler)
                st.write("Features extraites :")
                st.dataframe(input_df.T)
                st.metric("Probabilité de chute > 20%", f"{proba:.2%}")
            except Exception as e:
                st.error(f"Erreur: {e}")

    elif page == "Alertes":
        st.title("🚨 Alertes actives")
        try:
            df = pd.read_csv("log_predictions.csv", names=["timestamp", "symbol", "probability"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            alerts = df[df['probability'] > 0.7].sort_values("timestamp", ascending=False)
            st.dataframe(alerts.head(20))
        except Exception as e:
            st.error("Erreur de chargement : " + str(e))

# ---------------------- ▶️ MAIN ---------------------- #
if __name__ == '__main__':
    launch_dashboard()

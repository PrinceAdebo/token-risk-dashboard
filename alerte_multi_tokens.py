
import requests
import pandas as pd
import numpy as np
import asyncio

from ccxt.static_dependencies.marshmallow.utils import timestamp
from telegram import Bot
from datetime import datetime

# Configuration Telegram
TELEGRAM_TOKEN = "7501384463:AAHVKsgPjdrtN43AvPmAyrJNmBmZlle42zY"
CHAT_ID = "328665786"
bot = Bot(token=TELEGRAM_TOKEN)

TOKENS = ["OMUSDT", "SOLUSDT", "ETHEUSDT", "XRPUSDT"]
ALERT_LOG = "alert_log.csv"
csv_file = "data.csv"
pending_alerts = []

def send_alert(message):
    pending_alerts.append(message)

async def flush_alerts():
    for msg in pending_alerts:
        await bot.send_message(chat_id=CHAT_ID, text=msg)

def get_klines(symbol="OMUSDT", interval="15m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df

def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def is_hammer(open_price, high, low, close):
    body = abs(close - open_price)
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low
    return lower_shadow > 2 * body and upper_shadow < body

def log_alert(symbol, signal_type, rsi_value, price):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[now, symbol, signal_type, rsi_value, price]],
                         columns=["Date", "Token", "Signal", "RSI", "Price"])
    try:
        old = pd.read_csv(ALERT_LOG)
        combined = pd.concat([old, entry])
    except FileNotFoundError:
        combined = entry
    combined.to_csv(ALERT_LOG, index=False)

def is_false_signal(rsi_series):
    recent = rsi_series.dropna()[-5:]
    variation = recent.max() - recent.min()
    return variation < 5  # très peu de mouvement => possible faux signal
def write_csv(rows):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Token", "Signal", "RSI", "Price"])
        for row in rows:
            writer.writerow(row)

def analyze_all_tokens():
    log_rows = []
    for token in TOKENS:
        try:
            df = get_klines(symbol=token)
            closes = df['close'].values
            rsi_series = compute_rsi(closes)

            if len(rsi_series.dropna()) == 0:
                continue

            last_rsi = rsi_series.dropna().iloc[-1]
            last = df.iloc[-1]
            price = last['close']

            if is_false_signal(rsi_series):
                print(f"[Filtré] Faux signal potentiel détecté pour {token}")
                continue

            if last_rsi < 30: # and is_hammer(last['open'], last['high'], last['low'], last['close']):
                signal = "RSI < 30 (achat)"
                log_rows.append([timestamp, token, signal, f"{last_rsi:.2f}", f"{price:.4f}"])
                msg = (
                    "--------------------------\n"
                    f"🚨 ACHAT {token}\n"
                     f"🚨 Prix actuel {price}\n"
                    f"RSI: {last_rsi:.2f} < 30\n"
                    f"⚠️ Chandelier non vérifié\n"
                    "📈 Alerte RSI uniquement\n"
                    "--------------------------"
                )
                send_alert(msg)
                log_alert(token, "RSI < 30 + marteau", last_rsi, price)

            elif last_rsi > 70:
                signal = "RSI > 70 (vente)"
                log_rows.append([timestamp, token, signal, f"{last_rsi:.2f}", f"{price:.4f}"])
                msg = (
                    "--------------------------\n"
                    f"⚠️ VENTE {token}\nRSI: {last_rsi:.2f} > 70\n📉 Surachat → Correction possible."

                    "--------------------------"
                )
            elif last_rsi < 30 and is_hammer(last['open'], last['high'], last['low'], last['close']):
                signal = "RSI < 30 (achat)"
                log_rows.append([timestamp, token, signal, f"{last_rsi:.2f}", f"{price:.4f}"])
                msg = (
                    "--------------------------\n"
                    f"🚨 ACHAT {token}\n"
                    f"🚨 Prix actuel {price}\n"
                    f"RSI: {last_rsi:.2f} < 30\n"
                    f" Chandelier  vérifié\n"
                    "--------------------------"
                )


                send_alert(msg)
                log_alert(token, "RSI > 70", last_rsi, price)

        except Exception as e:
            print(f"Erreur avec {token}: {e}")
    if log_rows:
       write_csv(log_rows)


import requests

def get_market_sentiment():
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url)
        data = response.json()

        market_cap_change = data["data"]["market_cap_change_percentage_24h_usd"]
        btc_dominance = data["data"]["market_cap_percentage"]["btc"]

        if market_cap_change > 1:
            sentiment = "📈 Sentiment haussier"
        elif market_cap_change < -1:
            sentiment = "📉 Sentiment baissier"
        else:
            sentiment = "⚖️ Sentiment neutre"

        return f"{sentiment} | BTC Dominance: {btc_dominance:.1f}%", market_cap_change

    except Exception as e:
        return "❓ Sentiment inconnu", 0



# Exécution avec sentiment global
sentiment_text, _ = get_market_sentiment()
send_alert(f"🧠 Sentiment du marché : {sentiment_text}")

analyze_all_tokens()

if pending_alerts:
    asyncio.run(flush_alerts())

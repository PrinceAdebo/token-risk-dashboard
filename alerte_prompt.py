import requests
import pandas as pd
import numpy as np
from telegram import Bot
import asyncio

# Configuration
TELEGRAM_TOKEN = "7501384463:AAHVKsgPjdrtN43AvPmAyrJNmBmZlle42zY"
CHAT_ID = "328665786"
bot = Bot(token=TELEGRAM_TOKEN)

def send_alert(message):
    asyncio.run(bot.send_message(chat_id=CHAT_ID, text=message))

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

def analyze_and_alert():
    df = get_klines()
    closes = df['close'].values
    rsi_series = compute_rsi(closes)

    if len(rsi_series.dropna()) == 0:
        print("⛔ Pas assez de données pour calculer le RSI")
        return

    print("RSI disponible :", rsi_series.dropna().tail())
    send_alert("📊 RSI disponible :\n" + str(rsi_series.dropna().tail()))

    last_rsi = rsi_series.dropna().iloc[-1]

    last = df.iloc[-1]
    if last_rsi < 30 and is_hammer(last['open'], last['high'], last['low'], last['close']):
        msg = f"🚨 ALERTE PROMPTUSDT\nRSI: {last_rsi:.2f} < 30\nChandelier marteau détecté ✅\n⏳ Potentiel de rebond court terme."
        send_alert(msg)

analyze_and_alert()

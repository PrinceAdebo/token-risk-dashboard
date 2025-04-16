import ccxt
import requests
import time
import pandas as pd

# -------------------------------------------------------------------------
# 1. Paramètres de configuration
# -------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = "7739402863:AAFca0Z6usv8ZvAEc_qLFLaRmcKs6tE6z1A"
TELEGRAM_CHAT_ID = "328665786"

EXCHANGE_ID = "binance"  # ou un autre exchange CCXT, ex : 'coinbasepro', 'kraken'...
SYMBOL = "BTC/USDT"  # Remplacez par le symbole que vous souhaitez analyser
TIMEFRAME = "15m"  # Unité de temps : 15 minutes

# -------------------------------------------------------------------------
# 2. Initialiser l'exchange CCXT
# -------------------------------------------------------------------------
exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({
    "enableRateLimit": True,
})


# -------------------------------------------------------------------------
# 3. Fonctions pour la détection des patterns
# -------------------------------------------------------------------------
def is_hanging_man(open_price, high_price, low_price, close_price):
    """
    Détermine si une bougie correspond à un Hanging Man (Homme pendu).
    Règles simplifiées :
    - La bougie apparaît dans une phase haussière (à vous de vérifier le contexte).
    - Le corps est relativement petit, situé en haut de la bougie.
    - L'ombre (mèche) basse est au moins 2x la taille du corps.
    """
    body = abs(close_price - open_price)
    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price

    # Pour simplifier, on pose quelques conditions :
    # 1) Corps petit : body < (20% de la totalité de la bougie par exemple)
    # 2) Ombre basse >= 2 x body
    full_range = high_price - low_price
    if full_range == 0:
        return False

    body_ratio = body / full_range
    hanging_man_condition = (body_ratio < 0.2) and (lower_shadow >= 2 * body) and (upper_shadow < body)

    return hanging_man_condition


def is_doji(open_price, high_price, low_price, close_price):
    """
    Détermine si une bougie est un Doji.
    Règles simplifiées :
    - La différence entre open et close est très faible (ex : < 5% de la taille de la bougie).
    """
    body = abs(close_price - open_price)
    full_range = high_price - low_price
    if full_range == 0:
        return False

    # Par exemple, si le corps est < 5% de la bougie totale, on peut considérer que c'est un doji.
    ratio = body / full_range
    return (ratio < 0.05)


# -------------------------------------------------------------------------
# 4. Fonction d'envoi de notification Telegram
# -------------------------------------------------------------------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erreur lors de l'envoi du message Telegram : {e}")


# -------------------------------------------------------------------------
# 5. Récupération et analyse des bougies
# -------------------------------------------------------------------------
def analyze_candles():
    """
    Récupère les bougies en 15m pour la paire SYMBOL,
    vérifie la dernière bougie (ou les dernières) pour un Hanging Man
    ou un Doji.
    """
    # Récupération des bougies : 100 dernières bougies par ex.
    # ccxt renvoie des bougies sous la forme : [timestamp, open, high, low, close, volume]
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)

    # Convertir en DataFrame pour faciliter la lecture
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # On s'intéresse à la dernière bougie (ligne df.iloc[-1])
    last_candle = df.iloc[-1]
    open_price = last_candle["open"]
    high_price = last_candle["high"]
    low_price = last_candle["low"]
    close_price = last_candle["close"]

    # Vérifier si c'est un Hanging Man
    if is_hanging_man(open_price, high_price, low_price, close_price):
        message = (
            f"Hanging Man détecté sur {SYMBOL} (bougie 15mn)\n"
            f"Ouv: {open_price}, Haut: {high_price}, Bas: {low_price}, Close: {close_price}\n"
            f"Timestamp: {last_candle['timestamp']}"
        )
        send_telegram_message(message)

    # Vérifier si c'est un Doji
    if is_doji(open_price, high_price, low_price, close_price):
        message = (
            f"Doji détecté sur {SYMBOL} (bougie 15mn)\n"
            f"Ouv: {open_price}, Haut: {high_price}, Bas: {low_price}, Close: {close_price}\n"
            f"Timestamp: {last_candle['timestamp']}"
        )
        send_telegram_message(message)


# -------------------------------------------------------------------------
# 6. Boucle principale (run bot)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Bot démarré...")

    # Par exemple, on exécute la fonction toutes les 15 minutes
    # Vous pouvez utiliser un scheduler plus avancé (APScheduler, etc.).
    while True:
        try:
            analyze_candles()
        except Exception as e:
            print(f"Erreur lors de l'analyse : {e}")

        # Attendre 15 minutes (900 secondes) avant la prochaine vérification
        time.sleep(900)

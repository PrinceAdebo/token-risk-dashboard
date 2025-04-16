# extract_token_fundamentals.py
import requests
import pandas as pd
import time

# Fonction pour chercher l'identifiant CoinGecko d'un token
def get_token_id(symbol):
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    try:
        coins = response.json()
        if not isinstance(coins, list):
            print(f"❌ Résultat inattendu pour symbol '{symbol}':", coins)
            return None
    except Exception as e:
        print(f"❌ Erreur lors de l'appel API CoinGecko : {e}")
        return None

    for coin in coins:
        if coin['symbol'].lower() == symbol.lower():
            return coin['id']
    return None

# Récupère la market cap, secteur, followers Twitter pour un token donné
def get_token_fundamentals(coingecko_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    market_cap = data.get("market_data", {}).get("market_cap", {}).get("usd", None)
    categories = data.get("categories", [])
    community = data.get("community_data", {})
    twitter_followers = community.get("twitter_followers", None)
    telegram_user_count = community.get("telegram_channel_user_count", None)

    return {
        "coingecko_id": coingecko_id,
        "market_cap_usd": market_cap,
        "category": categories[0] if categories else None,
        "twitter_followers": twitter_followers,
        "telegram_users": telegram_user_count
    }

# Exemple d'usage avec liste de tokens
if __name__ == "__main__":
    token_symbols = ["om", "prompt", "baby", "stone", "fhe"]
    enriched_data = []

    for symbol in token_symbols:
        print(f"🔎 Recherche de {symbol}...")
        coingecko_id = get_token_id(symbol)
        if not coingecko_id:
            print(f"❌ ID introuvable pour {symbol}")
            continue
        time.sleep(1)
        fundamentals = get_token_fundamentals(coingecko_id)
        if fundamentals:
            fundamentals['symbol'] = symbol.upper()
            enriched_data.append(fundamentals)
        time.sleep(1)

    df = pd.DataFrame(enriched_data)
    df.to_csv("token_fundamentals.csv", index=False)
    print("✅ Fichier token_fundamentals.csv créé")

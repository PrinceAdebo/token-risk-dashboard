# merge_with_fundamentals.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger les fichiers
market_df = pd.read_csv("token_ai_dataset.csv")
fundamentals_df = pd.read_csv("token_fundamentals.csv")

# Mettre en majuscules les symboles pour correspondance
market_df["symbol"] = market_df["symbol"].str.upper()
fundamentals_df["symbol"] = fundamentals_df["symbol"].str.upper()

# Fusion des datasets
merged_df = pd.merge(market_df, fundamentals_df, on="symbol", how="left")

# Encodage des catégories textuelles (ex: secteur)
if "category" in merged_df.columns:
    le = LabelEncoder()
    merged_df["category_encoded"] = le.fit_transform(merged_df["category"].fillna("Unknown"))

# Remplacement des valeurs manquantes par 0 (ou autre stratégie selon ton cas)
merged_df["market_cap_usd"] = merged_df["market_cap_usd"].fillna(0)
merged_df["twitter_followers"] = merged_df["twitter_followers"].fillna(0)
merged_df["telegram_users"] = merged_df["telegram_users"].fillna(0)

# Export final
merged_df.to_csv("token_ai_dataset_enriched.csv", index=False)
print("✅ Dataset enrichi enregistré dans token_ai_dataset_enriched.csv")

# train_model_enriched.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Charger le dataset enrichi
df = pd.read_csv("token_ai_dataset_enriched.csv")

# Définir les features enrichies
features = [
    "open", "close", "volume", "rsi",
    "variation_1h", "variation_6h", "variation_24h",
    "market_cap_usd", "twitter_followers", "telegram_users", "category_encoded"
]

X = df[features]
y = df["drawdown_future_24h"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE : {mae:.2f}%")
print(f"✅ R² : {r2:.2f}")

# Importance des features
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Importance des variables (XGBoost avec fondamentaux)")
plt.tight_layout()
plt.savefig("xgb_feature_importance_enriched.png")
print("📊 Graphe enregistré dans xgb_feature_importance_enriched.png")

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("token_ai_dataset.csv")

# Features et target
features = ["open", "close", "volume", "rsi", "variation_1h", "variation_6h", "variation_24h"]
X = df[features]
y = df["drawdown_future_24h"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle XGBoost optimisé (rapide)
model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, verbosity=0)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE (erreur moyenne) : {mae:.2f}%")
print(f"✅ R² (qualité du modèle) : {r2:.2f}")

# Importance des variables
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.title("Importance des variables (XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
print("📊 Graphique enregistré dans xgb_feature_importance.png")

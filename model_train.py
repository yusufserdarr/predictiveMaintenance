# model_train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# ======= AYARLAR =======
DATA_CSV = "train_rul.csv"
FEATURES_TXT = "selected_features.txt"
MODEL_PKL = "model.pkl"
SCALER_PKL = "scaler.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =======================

# 1) Veriyi yükle
df = pd.read_csv(DATA_CSV)

# 2) Seçili özellikleri oku
with open(FEATURES_TXT, "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

X = df[selected_features]
y = df["RUL"]

# 3 Train-test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# 4 Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5 Model eğitimi
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)
model.fit(X_train_scaled, y_train)

# 6 Tahmin ve değerlendirme
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model Eğitildi")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.3f}")

# 7 Model ve scaler kaydet
joblib.dump(model, MODEL_PKL)
joblib.dump(scaler, SCALER_PKL)
print(f"💾 Model {MODEL_PKL} olarak kaydedildi.")
print(f"💾 Scaler {SCALER_PKL} olarak kaydedildi.")

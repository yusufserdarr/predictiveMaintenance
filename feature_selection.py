# feature_selection.py
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ======= AYARLAR =======
INPUT_CSV = "train_rul.csv"
TOP_K = 10                   # Kaç özelliği seçelim?
OUT_TXT = "selected_features.txt"
OUT_PNG = "feature_importance.png"
RANDOM_STATE = 42
# =======================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} bulunamadı. Önce rul_generator.py'yi çalıştırmalısın.")

    df = pd.read_csv(INPUT_CSV)

    # Hedef ve özellik kolonları
    target_col = "RUL"
    drop_cols = ["unit_number", "time_in_cycles", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].values

    # Ölçekleme (ağaç tabanlı modeller şart koşmaz; ama MI için faydalı)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # 1) RandomForest ile önem skoru
    rf = RandomForestRegressor(
        n_estimators=300, 
        random_state=RANDOM_STATE, 
        n_jobs=-1,
        min_samples_leaf=1,
        max_features='sqrt'
    )
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    # 2) Mutual Information (non-linear ilişkiyi de yakalar)
    mi_scores = mutual_info_regression(x_scaled, y, random_state=RANDOM_STATE)

    # Skorları birleştir (normalize edip ortalamasını alalım)
    rf_norm = (rf_importances - rf_importances.min()) / (np.ptp(rf_importances) + 1e-12)
    mi_norm = (mi_scores - np.min(mi_scores)) / (np.ptp(mi_scores) + 1e-12)

    blended = 0.6 * rf_norm + 0.4 * mi_norm  # ağırlık: RF 0.6, MI 0.4

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importances,
        "mi_score": mi_scores,
        "blended_score": blended
    }).sort_values("blended_score", ascending=False)

    topk = imp_df.head(TOP_K)["feature"].tolist()

    # Sonuçları yazdır
    print("\n✅ En önemli özellikler (TOP_K =", TOP_K, "):")
    for i, f in enumerate(topk, 1):
        print(f"{i:2d}. {f}")

    # TXT olarak kaydet
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for feat in topk:
            f.write(feat + "\n")
    print(f"\n💾 Seçilen özellikler {OUT_TXT} dosyasına kaydedildi.")

    # Grafiği kaydet
    plot_df = imp_df.head(TOP_K).sort_values("blended_score", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature"], plot_df["blended_score"])
    plt.title("Özellik Önem Sıralaması (RF + MI Birleşik Skor)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=100)
    plt.close()
    print(f"🖼️  {OUT_PNG} kaydedildi.")

if __name__ == "__main__":
    main()

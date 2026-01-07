import json
import sys

nb_path = '/Users/yusufserdaroglu/Desktop/tez/model_comparison.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 18'i bul (Deep Learning eğitimi)
target_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = ''.join(cell.get('source', []))
        if 'Deep Learning Modelleri Eğitiliyor' in source_str and 'SEQUENCE_LENGTH = 30' in source_str:
            target_idx = i
            break

if target_idx is None:
    print("❌ Cell 18 bulunamadı!")
    sys.exit(1)

print(f"✅ Cell 18 bulundu (index {target_idx})")

# Yeni kod
new_source = [
    "# Deep Learning parametreleri\n",
    "SEQUENCE_LENGTH = 30\n",
    "EPOCHS = 50\n",
    "BATCH_SIZE = 256\n",
    "TEST_SIZE = 0.2\n",
    "RANDOM_STATE = 42\n",
    "\n",
    "# Callbacks - daha akıllı ayarlar\n",
    "early_stop = EarlyStopping(\n",
    "    monitor=\"val_loss\", \n",
    "    patience=15,  # Daha fazla sabır\n",
    "    restore_best_weights=True,\n",
    "    min_delta=0.001  # Minimum iyileşme\n",
    ")\n",
    "reduce_lr = ReduceLROnPlateau(\n",
    "    monitor=\"val_loss\", \n",
    "    factor=0.5, \n",
    "    patience=8,  # Daha fazla sabır\n",
    "    min_lr=1e-7\n",
    ")\n",
    "\n",
    "print(\"🧠 Deep Learning Modelleri Eğitiliyor...\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "for ds_name in [\"FD001\", \"FD002\", \"FD003\", \"FD004\"]:\n",
    "    print(f\"\\n🧠 Deep Learning - Dataset: {ds_name}\")\n",
    "    print(\"-\" * 80)\n",
    "    \n",
    "    # Veriyi hazırla\n",
    "    train_df = datasets[ds_name][\"train\"]\n",
    "    feature_cols = datasets[ds_name][\"features\"]\n",
    "    \n",
    "    # Sequence oluştur\n",
    "    print(\"   📦 Sequencelar oluşturuluyor...\")\n",
    "    X_seq, y_seq = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)\n",
    "    print(f\"      ✅ {len(X_seq)} sequence oluşturuldu\")\n",
    "    \n",
    "    # Train-test split\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X_seq, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE\n",
    "    )\n",
    "    \n",
    "    # Scaling\n",
    "    scaler = StandardScaler()\n",
    "    # Reshape for scaling (samples, time_steps, features) -> (samples*time_steps, features)\n",
    "    X_train_reshape = X_train.reshape(-1, X_train.shape[-1])\n",
    "    X_train_scaled = scaler.fit_transform(X_train_reshape)\n",
    "    X_train_scaled = X_train_scaled.reshape(X_train.shape)\n",
    "    \n",
    "    X_test_reshape = X_test.reshape(-1, X_test.shape[-1])\n",
    "    X_test_scaled = scaler.transform(X_test_reshape)\n",
    "    X_test_scaled = X_test_scaled.reshape(X_test.shape)\n",
    "    \n",
    "    # Input shape definition\n",
    "    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])\n",
    "    print(f\"      ✅ Input shape: {input_shape}\")\n",
    "    \n",
    "    # Deep Learning modelleri\n",
    "    dl_models = {\n",
    "        \"LSTM\": create_lstm_model(input_shape),\n",
    "        \"GRU\": create_gru_model(input_shape),\n",
    "        \"CNN-LSTM\": create_cnn_lstm_model(input_shape),\n",
    "        \"Evolved-LSTM\": create_evolved_lstm(input_shape),\n",
    "        \"Evolved-GRU\": create_evolved_gru(input_shape)\n",
    "    }\n",
    "    \n",
    "    for model_name, model in dl_models.items():\n",
    "        print(f\"\\n   🔧 {model_name} eğitiliyor...\")\n",
    "        sys.stdout.flush()\n",
    "        \n",
    "        try:\n",
    "            start_time = time.time()\n",
    "            \n",
    "            # Model eğitimi - verbose=1 ile ilerleme göster\n",
    "            history = model.fit(\n",
    "                X_train_scaled, y_train,\n",
    "                epochs=EPOCHS,\n",
    "                batch_size=BATCH_SIZE,\n",
    "                validation_split=0.2,\n",
    "                callbacks=[early_stop, reduce_lr],\n",
    "                verbose=1  # İlerleme göster\n",
    "            )\n",
    "            \n",
    "            train_time = time.time() - start_time\n",
    "            \n",
    "            # Epoch kontrolü\n",
    "            epochs_trained = len(history.history[\"loss\"])\n",
    "            if epochs_trained == 0:\n",
    "                print(f\"      ❌ HATA: {model_name} hiç epoch çalıştırmadı!\")\n",
    "                continue\n",
    "            \n",
    "            if train_time < 1.0:\n",
    "                print(f\"      ⚠️  UYARI: {model_name} eğitim süresi çok kısa ({train_time:.3f}s) - kontrol edin!\")\n",
    "            \n",
    "            # Tahmin\n",
    "            y_pred = model.predict(X_test_scaled, verbose=0).flatten()\n",
    "            \n",
    "            # Değerlendirme\n",
    "            mae = mean_absolute_error(y_test, y_pred)\n",
    "            rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "            r2 = r2_score(y_test, y_pred)\n",
    "            \n",
    "            # MAPE\n",
    "            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100\n",
    "            \n",
    "            # Accuracy\n",
    "            accuracy_10 = np.mean(np.abs(y_test - y_pred) <= 10) * 100\n",
    "            accuracy_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100\n",
    "            \n",
    "            all_results.append({\n",
    "                \"Model\": model_name,\n",
    "                \"Dataset\": ds_name,\n",
    "                \"MAE\": mae,\n",
    "                \"RMSE\": rmse,\n",
    "                \"R2\": r2,\n",
    "                \"MAPE\": mape,\n",
    "                \"Accuracy_5\": accuracy_5,\n",
    "                \"Accuracy_10\": accuracy_10,\n",
    "                \"Train_Time\": train_time,\n",
    "                \"Epochs\": epochs_trained\n",
    "            })\n",
    "            \n",
    "            print(f\"      ✅ MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f} | MAPE: {mape:.2f}% | Acc(±5): {accuracy_5:.1f}% | Epochs: {epochs_trained} | Time: {train_time:.2f}s\")\n",
    "            sys.stdout.flush()\n",
    "            \n",
    "        except Exception as e:\n",
    "            print(f\"      ❌ HATA: {model_name} eğitilirken hata: {type(e).__name__}: {e}\")\n",
    "            import traceback\n",
    "            traceback.print_exc()\n",
    "            continue\n",
    "\n",
    "print(\"\\n✅ Tüm Deep Learning eğitimleri tamamlandı!\")\n"
]

# Cell'i güncelle
nb['cells'][target_idx]['source'] = new_source
nb['cells'][target_idx]['outputs'] = []  # Eski çıktıları temizle

# Kaydet
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Cell 18 düzeltildi!")
print("   - verbose=1 eklendi (ilerleme gösterir)")
print("   - Hata yakalama eklendi")
print("   - Epoch kontrolü eklendi")
print("   - Early stopping patience artırıldı (15)")



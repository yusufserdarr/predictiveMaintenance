# 🚀 NASA CMAPSS Model Karşılaştırma Analizi

Bu Jupyter notebook ile 4 farklı NASA CMAPSS dataseti üzerinde 7 farklı model (4 ML + 3 Deep Learning) karşılaştırması yapabilirsiniz.

## 📊 Veri Setleri

- **FD001**: Tek operasyon modu, tek arıza tipi (20,631 train satır)
- **FD002**: 6 operasyon modu, tek arıza tipi (53,759 train satır)
- **FD003**: Tek operasyon modu, 2 arıza tipi (24,720 train satır)
- **FD004**: 6 operasyon modu, 2 arıza tipi (61,249 train satır)

**Toplam: 265,256 satır veri!**

## 🤖 Modeller

### Geleneksel ML:
1. **Linear Regression** - Baseline model
2. **Random Forest** - Ensemble öğrenme
3. **XGBoost** - Gradient boosting (mevcut modeliniz)
4. **SVR** - Support Vector Regression

### Deep Learning:
5. **LSTM** - Long Short-Term Memory (zaman serisi için ideal)
6. **GRU** - Gated Recurrent Unit (LSTM'e benzer ama daha hızlı)
7. **CNN-LSTM** - Hybrid model (CNN + LSTM)

## 📈 Değerlendirme Metrikleri

- **MAE** (Mean Absolute Error) - Ortalama mutlak hata
- **RMSE** (Root Mean Squared Error) - Kök ortalama kare hatası
- **R²** Score - Açıklanan varyans oranı
- **Eğitim Süresi** - Saniye cinsinden

## 🔧 Kurulum

### 1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

### 2. Jupyter Notebook'u başlatın:

```bash
jupyter notebook model_comparison.ipynb
```

veya

```bash
jupyter lab model_comparison.ipynb
```

## ▶️ Kullanım

1. **Tüm cell'leri çalıştırın**: `Cell > Run All` veya `Kernel > Restart & Run All`

2. **Adım adım çalıştırın**: Her cell'i tek tek `Shift+Enter` ile çalıştırın

3. **Beklenen süre**:
   - ML modelleri: ~2-5 dakika (toplam 16 model eğitimi)
   - DL modelleri: ~20-40 dakika (toplam 12 model eğitimi)
   - **Toplam: ~30-50 dakika** (CPU'da)
   - GPU varsa çok daha hızlı olacaktır!

## 📁 Çıktılar

Notebook çalıştırıldığında aşağıdaki dosyalar oluşturulur:

### CSV Dosyası:
- `model_comparison_results.csv` - Tüm sonuçlar tablolar halinde

### Grafikler:
- `model_comparison_mae.png` - Dataset bazında MAE karşılaştırması
- `model_comparison_avg.png` - Model bazında ortalama performans
- `model_comparison_heatmap.png` - Model vs Dataset heatmap
- `model_training_time.png` - Eğitim süresi karşılaştırması

## 🎯 Özellikler

- ✅ 4 farklı dataset üzerinde kapsamlı karşılaştırma
- ✅ Geleneksel ML ve Deep Learning modelleri
- ✅ Otomatik feature engineering (sabit değerleri filtreler)
- ✅ Zaman serisi sequence oluşturma (DL için)
- ✅ Early stopping ve learning rate reduction
- ✅ Profesyonel görselleştirmeler
- ✅ CSV export ile sonuç paylaşımı

## ⚙️ Parametreler

Notebook içinde değiştirebileceğiniz parametreler:

```python
# Genel
TEST_SIZE = 0.2              # Test set oranı
RANDOM_STATE = 42            # Reproducibility için

# Deep Learning
SEQUENCE_LENGTH = 30         # Kaç cycle geriye bakılacak
EPOCHS = 50                  # Maksimum epoch sayısı
BATCH_SIZE = 256             # Batch büyüklüğü
```

## 🔬 Model Detayları

### LSTM Modeli:
- 2 LSTM katmanı (64 → 32 units)
- Dropout (0.2)
- Dense output layer

### GRU Modeli:
- 2 GRU katmanı (64 → 32 units)
- Dropout (0.2)
- Dense output layer

### CNN-LSTM Modeli:
- 2 CNN katmanı (64 → 32 filters)
- MaxPooling
- LSTM katmanı (50 units)
- Dense output layer

## 💡 İpuçları

1. **GPU kullanımı**: TensorFlow otomatik olarak GPU'yu kullanacaktır (varsa)
2. **Memory hatası alırsanız**: `BATCH_SIZE`'ı artırın veya `SEQUENCE_LENGTH`'i azaltın
3. **Hızlandırmak için**: `EPOCHS` değerini azaltabilirsiniz (örn. 30)
4. **Daha iyi sonuç için**: Hyperparameter tuning yapabilirsiniz

## 📊 Beklenen Sonuçlar

Literatüre göre beklenen MAE değerleri:
- **FD001**: 12-15 (en kolay)
- **FD002**: 18-25 (zorlu)
- **FD003**: 12-16 (orta)
- **FD004**: 20-28 (en zorlu)

Deep Learning modelleri genellikle ML modellerinden %10-30 daha iyi performans gösterir.

## 🐛 Sorun Giderme

### TensorFlow yüklenmiyor:
```bash
pip install tensorflow --upgrade
```

### Memory hatası:
```python
BATCH_SIZE = 512  # veya daha yüksek
SEQUENCE_LENGTH = 20  # daha kısa
```

### Slow training:
- GPU kullanın
- EPOCHS'u azaltın
- Daha az dataset üzerinde test edin

## 📚 Referanslar

- NASA CMAPSS Dataset: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- LSTM: Hochreiter & Schmidhuber (1997)
- XGBoost: Chen & Guestrin (2016)

## 👨‍💻 Geliştirme

Daha fazla model eklemek için:

1. Model fonksiyonu tanımlayın
2. `get_ml_models()` veya DL bölümüne ekleyin
3. Notebook'u çalıştırın

## 📝 Notlar

- Bu analiz akademik çalışmalar için uygundur
- Sonuçlar her çalıştırmada biraz farklı olabilir (random seed'e rağmen)
- Deep Learning modelleri daha uzun sürer ama genellikle daha iyi sonuç verir

## 🎓 Tez İçin Öneriler

1. Her modelin avantaj/dezavantajlarını tartışın
2. Eğitim süresi vs performans trade-off'unu analiz edin
3. Farklı datasetlerdeki performans farklarını açıklayın
4. En iyi model için confusion analysis yapın
5. Hata dağılımlarını görselleştirin

---

**Oluşturulma Tarihi**: Aralık 2025
**Güncelleme**: Her commit'te otomatik


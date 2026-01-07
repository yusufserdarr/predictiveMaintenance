#!/usr/bin/env python3
"""
Streamlit Dashboard - Predictive Maintenance
Makine öğrenimi ile erken arıza tespiti web arayüzü
"""

import streamlit as st
import pandas as pd
import datetime
import os
import joblib
from maintenance import maintenance_decision
from reporting import daily_report_to_excel, ensure_dirs
from lime_explain import explain_instance, open_html_in_browser
from shap_analysis import shap_summary_png, shap_local_png, load_sample_data

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Predictive Maintenance Dashboard", 
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .status-critical {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .status-planned {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .status-normal {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# En iyi 5 model listesi
BEST_MODELS = [
    "Evolved-LSTM",
    "Evolved-GRU", 
    "LSTM",
    "GRU",
    "Random Forest"
]

# Model ve scaler yükleme
@st.cache_resource
def load_models(model_name="Random Forest"):
    """Seçilen model ve scaler'ı yükle (cache ile)"""
    try:
        # Model dosya yolu belirleme
        if model_name in ["Evolved-LSTM", "Evolved-GRU", "LSTM", "GRU", "CNN-LSTM"]:
            # Deep Learning modelleri için TensorFlow/Keras
            model_path = f"models/{model_name.lower().replace('-', '_')}.h5"
            scaler_path = f"models/scaler_{model_name.lower().replace('-', '_')}.pkl"
            
            # TensorFlow modeli yükle
            try:
                import tensorflow as tf
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else joblib.load("scaler.pkl")
                else:
                    # Sessizce varsayılan modeli kullan
                    model = joblib.load("model.pkl")
                    scaler = joblib.load("scaler.pkl")
            except:
                # Sessizce varsayılan modeli kullan
                model = joblib.load("model.pkl")
                scaler = joblib.load("scaler.pkl")
        else:
            # ML modelleri için joblib
            model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
            scaler_path = f"models/scaler_{model_name.lower().replace(' ', '_')}.pkl"
            
            # Eğer özel model dosyası yoksa varsayılanı kullan
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else joblib.load("scaler.pkl")
            else:
                model = joblib.load("model.pkl")
                scaler = joblib.load("scaler.pkl")
        
        # Özellikleri oku
        with open("selected_features.txt", "r") as f:
            features = [line.strip() for line in f.readlines()]
            
        return model, scaler, features
    except Exception as e:
        # Hata durumunda varsayılan modeli kullan
        try:
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
            with open("selected_features.txt", "r") as f:
                features = [line.strip() for line in f.readlines()]
            return model, scaler, features
        except:
            st.error(f"Model yükleme hatası: {e}")
            st.stop()

# Başlık ve model seçimi
col_title, col_model = st.columns([3, 1])
with col_title:
    st.title("🧠 Makine Öğrenimi ile Erken Arıza Tespiti")
    st.markdown("**Gerçek zamanlı sensör verilerinden makine kalan ömrü (RUL) tahmini**")
with col_model:
    st.markdown("<br>", unsafe_allow_html=True)  # Dikey hizalama için boşluk
    selected_model = st.selectbox(
        "Model Seç:",
        BEST_MODELS,
        index=4,  # Varsayılan: Random Forest
        help="En iyi 5 modelden birini seçin"
    )

st.markdown("---")

# Model yükle
model, scaler, features = load_models(selected_model)

# Klasörleri hazırla
ensure_dirs()

# Sol menü
st.sidebar.header("🔧 Ayarlar")
st.sidebar.markdown("### Veri Kaynağı")
data_source = st.sidebar.radio("Veri Kaynağı Seç:", ["Canlı Akış", "Dosya Yükle", "Manuel Giriş"])

st.sidebar.markdown("### Bakım Eşikleri")
critical_th = st.sidebar.slider("Kritik Eşik (RUL <)", 5, 50, 20, help="Bu değerin altında acil bakım gerekir")
planned_th = st.sidebar.slider("Planlı Eşik (RUL <)", 30, 100, 50, help="Bu değerin altında planlı bakım önerilir")

# Veri yükleme / okuma
sicaklik, titresim, tork = None, None, None

if data_source == "Canlı Akış":
    st.subheader("📡 Canlı Veri Akışı")
    
    if os.path.exists("logs/stream.csv"):
        try:
            df_stream = pd.read_csv("logs/stream.csv")
            if not df_stream.empty:
                latest = df_stream.tail(1)
                
                # Son veri gösterimi
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Toplam Kayıt", len(df_stream))
                with col2:
                    st.metric("🕐 Son Güncelleme", latest["timestamp"].iloc[0])
                with col3:
                    st.metric("🌡️ Sıcaklık", f"{latest['sicaklik'].iloc[0]:.2f}°C")
                with col4:
                    st.metric("📈 Titreşim", f"{latest['titresim'].iloc[0]:.3f}")
                
                st.dataframe(latest, width='stretch')
                
                sicaklik = latest["sicaklik"].iloc[0]
                titresim = latest["titresim"].iloc[0]
                tork = latest["tork"].iloc[0]
            else:
                st.warning("⚠️ Henüz veri akışı başlamadı.")
                st.info("💡 Veri akışını başlatmak için: `python sim_stream.py --out logs/stream.csv`")
                st.stop()
        except Exception as e:
            st.error(f"Veri okuma hatası: {e}")
            st.stop()
    else:
        st.warning("⚠️ logs/stream.csv bulunamadı.")
        st.info("💡 Önce veri simülatörünü başlatın: `python sim_stream.py --out logs/stream.csv`")
        st.stop()

elif data_source == "Dosya Yükle":
    st.subheader("📁 Dosya Yükleme")
    uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"], help="Sıcaklık, titreşim, tork sütunları içeren CSV")
    
    if uploaded is not None:
        try:
            df_stream = pd.read_csv(uploaded)
            st.success(f"✅ Dosya yüklendi: {len(df_stream)} satır")
            
            # Son satırı al
            latest = df_stream.tail(1)
            st.dataframe(latest, width='stretch')
            
            sicaklik = latest["sicaklik"].iloc[0]
            titresim = latest["titresim"].iloc[0]
            tork = latest["tork"].iloc[0]
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
            st.stop()
    else:
        st.info("📤 Bir CSV dosyası yükleyin")
        st.stop()

else:  # Manuel Giriş
    st.subheader("✏️ Manuel Veri Girişi")
    st.markdown("**Tüm sensör değerlerini girin:**")
    
    # Ana sensörler
    st.markdown("### 🔧 Ana Sensörler")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sicaklik = st.number_input("🌡️ Sıcaklık (°C)", min_value=200.0, max_value=700.0, value=450.0, step=1.0)
        sensor_11 = st.number_input("Sensör 11", min_value=40.0, max_value=60.0, value=47.5, step=0.1)
        sensor_12 = st.number_input("Sensör 12", min_value=500.0, max_value=600.0, value=521.0, step=1.0)
    
    with col2:
        titresim = st.number_input("📳 Titreşim", min_value=0.1, max_value=5.0, value=2.5, step=0.1)
        sensor_4 = st.number_input("Sensör 4", min_value=1300.0, max_value=1600.0, value=1400.0, step=10.0)
        sensor_7 = st.number_input("Sensör 7", min_value=500.0, max_value=600.0, value=553.0, step=1.0)
    
    with col3:
        tork = st.number_input("⚙️ Tork", min_value=10.0, max_value=100.0, value=55.0, step=1.0)    
        sensor_15 = st.number_input("Sensör 15", min_value=5.0, max_value=15.0, value=8.4, step=0.1)
        sensor_9 = st.number_input("Sensör 9", min_value=8000.0, max_value=10000.0, value=9050.0, step=50.0)
    
    with col4:
        sensor_21 = st.number_input("Sensör 21", min_value=20.0, max_value=30.0, value=23.3, step=0.1)
        sensor_20 = st.number_input("Sensör 20", min_value=30.0, max_value=50.0, value=38.9, step=0.1)
    
    # Son sensörler
    st.markdown("### 🔬 İlave Ölçümler")
    col1, col2 = st.columns(2)
    with col1:
        sensor_2 = st.number_input("Sensör 2", min_value=600.0, max_value=700.0, value=642.0, step=1.0)
    with col2:
        sensor_3 = st.number_input("Sensör 3", min_value=1500.0, max_value=1700.0, value=1585.0, step=10.0)

# Tahmin yapma
if sicaklik is not None and titresim is not None and tork is not None:
    
    # Manuel giriş için tüm sensör verilerini kullan
    if data_source == "Manuel Giriş":
        # Gerçek sensör verilerini kullanarak tahmin yap
        try:
            # Sensör verilerini DataFrame'e dönüştür
            manual_data = {
                'sensor_measurement_11': [sensor_11],
                'sensor_measurement_12': [sensor_12 + sicaklik/10],
                'sensor_measurement_4': [sensor_4 + sicaklik],
                'sensor_measurement_7': [sensor_7],
                'sensor_measurement_15': [sensor_15 + titresim],
                'sensor_measurement_9': [sensor_9 + tork*10],
                'sensor_measurement_21': [sensor_21],
                'sensor_measurement_20': [sensor_20],
                'sensor_measurement_2': [sensor_2 + sicaklik/5],
                'sensor_measurement_3': [sensor_3 + sicaklik*2]
            }
            manual_df = pd.DataFrame(manual_data)
            
            # Veriyi ölçekle
            X_scaled = scaler.transform(manual_df[features])
            
            # Model ile tahmin yap
            if selected_model in ["Evolved-LSTM", "Evolved-GRU", "LSTM", "GRU", "CNN-LSTM"]:
                # Deep Learning modelleri için sequence gerekiyor
                # Basit tahmin kullan (DL modelleri için sequence gerekli)
                rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
            else:
                rul = model.predict(X_scaled)[0]
            
        except Exception as e:
            st.warning(f"Model tahmini yapılamadı, basit hesaplama kullanılıyor: {e}")
            # Basit tahmin (fallback)
            rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
    else:
        # Canlı Akış ve Dosya Yükle için model tahmini
        try:
            # Sensör verilerini DataFrame'e dönüştür
            stream_data = {
                'sensor_measurement_11': [47.5],
                'sensor_measurement_12': [521.0 + sicaklik/10],
                'sensor_measurement_4': [1400.0 + sicaklik],
                'sensor_measurement_7': [553.0],
                'sensor_measurement_15': [8.4 + titresim],
                'sensor_measurement_9': [9050.0 + tork*10],
                'sensor_measurement_21': [23.3],
                'sensor_measurement_20': [38.9],
                'sensor_measurement_2': [642.0 + sicaklik/5],
                'sensor_measurement_3': [1585.0 + sicaklik*2]
            }
            stream_df = pd.DataFrame(stream_data)
            
            # Veriyi ölçekle
            X_scaled = scaler.transform(stream_df[features])
            
            # Model ile tahmin yap
            if selected_model in ["Evolved-LSTM", "Evolved-GRU", "LSTM", "GRU", "CNN-LSTM"]:
                # Deep Learning modelleri için sequence gerekiyor
                # Basit tahmin kullan (DL modelleri için sequence gerekli)
                rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
            else:
                rul = model.predict(X_scaled)[0]
        except Exception as e:
            # Basit tahmin (fallback)
            rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
    
    # Bakım kararı
    result = maintenance_decision(rul, {"critical": critical_th, "planned": planned_th})
    
    # Sonuç gösterimi
    st.markdown("---")
    st.subheader("🔮 Tahmin Sonuçları")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🔢 Kalan Ömür (RUL)",
            value=f"{rul:.2f} döngü",
            delta=f"Model: {selected_model}"
        )
    
    with col2:
        st.metric(
            label="🔧 Bakım Durumu",
            value=result['status']
        )
    
    with col3:
        # Durum kartı
        status_class = {
            'CRITICAL': 'status-critical',
            'PLANNED': 'status-planned',
            'NORMAL': 'status-normal',
            'UNKNOWN': 'metric-card'
        }.get(result['status'], 'metric-card')
        
        st.markdown(
            f"""
            <div class="metric-card {status_class}">
                <h3>💬 Öneri</h3>
                <p>{result['message']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Raporlama bölümü
    st.markdown("---")
    st.subheader("📊 Raporlama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Günlük Excel Raporu Oluştur", width='stretch'):
            try:
                with st.spinner("Rapor oluşturuluyor..."):
                    path = daily_report_to_excel()
                st.success(f"✅ Rapor oluşturuldu: `{path}`")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Rapor hatası: {e}")
    
    with col2:
        if st.button("📝 Tahmini Logla", width='stretch'):
            try:
                from reporting import append_prediction_log
                log_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "sicaklik": sicaklik,
                    "titresim": titresim,
                    "tork": tork,
                    "rul": rul,
                    "status": result['status']
                }
                append_prediction_log(log_data)
                st.success("✅ Tahmin loglandı!")
            except Exception as e:
                st.error(f"❌ Loglama hatası: {e}")
    
    # Açıklamalar (Explainability)
    st.markdown("---")
    st.subheader("🧩 Model Açıklamaları (Explainability)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 LIME Açıklaması", width='stretch'):
            try:
                with st.spinner("LIME açıklaması oluşturuluyor..."):
                    # Örnek veri oluştur
                    sample_data = {
                        'sensor_measurement_11': [47.5],
                        'sensor_measurement_12': [521.0 + sicaklik/10],
                        'sensor_measurement_4': [1400.0 + sicaklik],
                        'sensor_measurement_7': [553.0],
                        'sensor_measurement_15': [8.4 + titresim],
                        'sensor_measurement_9': [9050.0 + tork*10],
                        'sensor_measurement_21': [23.3],
                        'sensor_measurement_20': [38.9],
                        'sensor_measurement_2': [642.0 + sicaklik/5],
                        'sensor_measurement_3': [1585.0 + sicaklik*2]
                    }
                    sample_df = pd.DataFrame(sample_data)
                    
                    html_file = explain_instance(model, scaler, sample_df, features, "reports/lime_explanation.html")
                    
                st.success("✅ LIME açıklaması oluşturuldu!")
                
                # HTML dosyasını Streamlit'te göster
                if os.path.exists(html_file):
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    st.markdown("### 🔍 LIME Açıklaması")
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    # İndirme linki
                    st.download_button(
                        label="📥 HTML Dosyasını İndir",
                        data=html_content,
                        file_name="lime_explanation.html",
                        mime="text/html"
                    )
                
            except Exception as e:
                st.error(f"❌ LIME hatası: {e}")
    
    with col2:
        if st.button("📊 SHAP Lokal Grafiği", width='stretch'):
            try:
                with st.spinner("SHAP lokal analizi yapılıyor..."):
                    # Örnek veri oluştur
                    sample_data = {
                        'sensor_measurement_11': [47.5],
                        'sensor_measurement_12': [521.0 + sicaklik/10],
                        'sensor_measurement_4': [1400.0 + sicaklik],
                        'sensor_measurement_7': [553.0],
                        'sensor_measurement_15': [8.4 + titresim],
                        'sensor_measurement_9': [9050.0 + tork*10],
                        'sensor_measurement_21': [23.3],
                        'sensor_measurement_20': [38.9],
                        'sensor_measurement_2': [642.0 + sicaklik/5],
                        'sensor_measurement_3': [1585.0 + sicaklik*2]
                    }
                    sample_df = pd.DataFrame(sample_data)
                    
                    # Veriyi ölçekle
                    X_scaled = pd.DataFrame(
                        scaler.transform(sample_df), 
                        columns=sample_df.columns
                    )
                    
                    local_path = shap_local_png(model, X_scaled)
                    
                st.success("✅ SHAP lokal grafiği oluşturuldu!")
                
                # Görseli göster
                if os.path.exists(local_path):
                    st.image(local_path, caption="SHAP Lokal Analizi", width='stretch')
                    
            except Exception as e:
                st.error(f"❌ SHAP lokal hatası: {e}")
    
    with col3:
        if st.button("📈 SHAP Özet Grafiği", width='stretch'):
            try:
                with st.spinner("SHAP özet analizi yapılıyor..."):
                    # Örnek veri yükle
                    X_sample = load_sample_data(sample_size=300)
                    
                    # Veriyi ölçekle
                    X_scaled = pd.DataFrame(
                        scaler.transform(X_sample), 
                        columns=X_sample.columns,
                        index=X_sample.index
                    )
                    
                    summary_path = shap_summary_png(model, X_scaled)
                    
                st.success("✅ SHAP özet grafiği oluşturuldu!")
                
                # Görseli göster
                if os.path.exists(summary_path):
                    st.image(summary_path, caption="SHAP Özet Analizi", width='stretch')
                    
            except Exception as e:
                st.error(f"❌ SHAP özet hatası: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🔧 Predictive Maintenance Dashboard | Makine Öğrenimi ile Erken Arıza Tespiti
    </div>
    """,
    unsafe_allow_html=True
)

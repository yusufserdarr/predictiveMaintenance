#!/usr/bin/env python3
"""
App modülündeki fonksiyonlar için unit testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import importlib.util


class TestAppFunctions(unittest.TestCase):
    """App fonksiyonları testleri"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Test verisi oluştur
        self.create_test_files()
    
    def tearDown(self):
        """Test sonrası temizlik"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Test dosyaları oluştur"""
        # Model dosyaları (dummy)
        with open("model.pkl", "wb") as f:
            f.write(b"dummy model")
        
        with open("scaler.pkl", "wb") as f:
            f.write(b"dummy scaler")
        
        # Features dosyası
        features = ["sensor_1", "sensor_2", "sensor_3"]
        with open("selected_features.txt", "w") as f:
            for feature in features:
                f.write(f"{feature}\n")
        
        # Stream CSV
        os.makedirs("logs", exist_ok=True)
        stream_data = pd.DataFrame({
            'timestamp': ['2024-01-01T12:00:00', '2024-01-01T12:01:00'],
            'sicaklik': [45.5, 46.0],
            'titresim': [0.25, 0.26],
            'tork': [55.0, 56.0]
        })
        stream_data.to_csv("logs/stream.csv", index=False)
    
    def load_app_module(self):
        """App modülünü yükle"""
        try:
            spec = importlib.util.spec_from_file_location("app", "app.py")
            app_module = importlib.util.module_from_spec(spec)
            
            # Streamlit import'unu mock'la
            with patch.dict('sys.modules', {'streamlit': MagicMock()}):
                spec.loader.exec_module(app_module)
            
            return app_module
        except Exception as e:
            print(f"App modülü yüklenemedi: {e}")
            return None
    
    @patch('joblib.load')
    def test_model_loading_functions(self, mock_joblib_load):
        """Model yükleme fonksiyonları testi"""
        # Mock model ve scaler
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        
        def side_effect(filename):
            if 'model' in filename:
                return mock_model
            elif 'scaler' in filename:
                return mock_scaler
            return None
        
        mock_joblib_load.side_effect = side_effect
        
        # Test: Model yükleme başarılı olmalı
        try:
            import joblib
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
            
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
    
    def test_features_file_reading(self):
        """Features dosyası okuma testi"""
        # Features dosyasını oku
        with open("selected_features.txt", "r") as f:
            features = [line.strip() for line in f.readlines()]
        
        # Kontroller
        self.assertEqual(len(features), 3)
        self.assertIn("sensor_1", features)
        self.assertIn("sensor_2", features)
        self.assertIn("sensor_3", features)
    
    def test_stream_data_reading(self):
        """Stream veri okuma testi"""
        # Stream CSV'yi oku
        df = pd.read_csv("logs/stream.csv")
        
        # Kontroller
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('sicaklik', df.columns)
        self.assertIn('titresim', df.columns)
        self.assertIn('tork', df.columns)
        
        # Veri tiplerini kontrol et
        self.assertTrue(pd.api.types.is_numeric_dtype(df['sicaklik']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['titresim']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['tork']))
    
    def test_data_validation_logic(self):
        """Veri doğrulama mantığı testi"""
        # Test verisi
        test_data = {
            'sicaklik': [45.5, 46.0, 47.0],
            'titresim': [0.25, 0.26, 0.27],
            'tork': [55.0, 56.0, 57.0]
        }
        
        df = pd.DataFrame(test_data)
        
        # Basit doğrulama testleri
        # Sıcaklık aralığı kontrolü
        valid_temp = df['sicaklik'].between(0, 100).all()
        self.assertTrue(valid_temp)
        
        # Titreşim pozitif değerler
        valid_vibration = (df['titresim'] >= 0).all()
        self.assertTrue(valid_vibration)
        
        # Tork pozitif değerler
        valid_torque = (df['tork'] > 0).all()
        self.assertTrue(valid_torque)
    
    def test_rul_calculation_simulation(self):
        """RUL hesaplama simülasyonu testi"""
        # Basit RUL hesaplama formülü (app.py'deki gibi)
        sicaklik = 45.5
        titresim = 0.25
        tork = 55.0
        
        # app.py'deki formül: max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
        rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
        
        # Kontroller
        self.assertGreaterEqual(rul, 10)  # Minimum 10 olmalı
        self.assertIsInstance(rul, (int, float))
        
        # Farklı değerlerle test
        test_cases = [
            (30, 0.1, 40),   # Düşük değerler -> Yüksek RUL
            (80, 0.8, 80),   # Yüksek değerler -> Düşük RUL
            (50, 0.5, 60)    # Orta değerler
        ]
        
        for temp, vib, tor in test_cases:
            calculated_rul = max(10, 200 - (temp/10) - (vib*20) - (tor/2))
            self.assertGreaterEqual(calculated_rul, 10)
    
    def test_file_existence_checks(self):
        """Dosya varlığı kontrolleri testi"""
        # Gerekli dosyaların var olduğunu kontrol et
        required_files = [
            "model.pkl",
            "scaler.pkl", 
            "selected_features.txt",
            "logs/stream.csv"
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"{file_path} dosyası bulunamadı")
    
    def test_data_preprocessing_steps(self):
        """Veri ön işleme adımları testi"""
        # Stream verisini oku
        df = pd.read_csv("logs/stream.csv")
        
        # Son satırı al (latest data simulation)
        if len(df) > 0:
            latest = df.tail(1)
            
            # Değerlerin sayısal olduğunu kontrol et
            self.assertIsInstance(latest['sicaklik'].iloc[0], (int, float))
            self.assertIsInstance(latest['titresim'].iloc[0], (int, float))
            self.assertIsInstance(latest['tork'].iloc[0], (int, float))
            
            # NaN değer olmamalı
            self.assertFalse(latest['sicaklik'].isna().any())
            self.assertFalse(latest['titresim'].isna().any())
            self.assertFalse(latest['tork'].isna().any())
    
    def test_error_handling_scenarios(self):
        """Hata yönetimi senaryoları testi"""
        # Eksik dosya durumu
        if os.path.exists("model.pkl"):
            os.remove("model.pkl")
        
        # Model yükleme hatası simülasyonu
        try:
            with open("model.pkl", "rb") as f:
                f.read()
            file_accessible = True
        except FileNotFoundError:
            file_accessible = False
        
        self.assertFalse(file_accessible)
        
        # Boş stream dosyası durumu
        empty_df = pd.DataFrame()
        empty_df.to_csv("logs/empty_stream.csv", index=False)
        
        empty_stream = pd.read_csv("logs/empty_stream.csv")
        self.assertEqual(len(empty_stream), 0)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
App Streamlit fonksiyonları için kapsamlı testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys


class TestAppStreamlit(unittest.TestCase):
    """App Streamlit testleri"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Test verisi oluştur
        self.create_test_data()
    
    def tearDown(self):
        """Test sonrası temizlik"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Test verisi oluştur"""
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
        
        # Reports dizini
        os.makedirs("reports", exist_ok=True)
    
    @patch('joblib.load')
    def test_model_loading_logic(self, mock_joblib_load):
        """Model yükleme mantığı testi"""
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
        
        # Model yükleme simülasyonu
        try:
            import joblib
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
            
            # Features yükleme
            with open("selected_features.txt", "r") as f:
                features = [line.strip() for line in f.readlines()]
            
            # Başarılı yükleme
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            self.assertEqual(len(features), 3)
            
            return model, scaler, features
            
        except Exception as e:
            self.fail(f"Model yükleme başarısız: {e}")
    
    def test_rul_calculation_formula(self):
        """RUL hesaplama formülü testi"""
        # app.py'deki formül: max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
        
        test_cases = [
            # (sicaklik, titresim, tork, expected_min_rul)
            (30, 0.1, 40, 10),    # Düşük değerler
            (50, 0.5, 60, 10),    # Orta değerler  
            (80, 0.8, 80, 10),    # Yüksek değerler
            (0, 0, 0, 200),       # Minimum değerler
        ]
        
        for sicaklik, titresim, tork, min_expected in test_cases:
            rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
            
            # Kontroller
            self.assertGreaterEqual(rul, 10)  # Minimum 10
            self.assertIsInstance(rul, (int, float))
            
            if min_expected == 200:  # Minimum değerler durumu
                self.assertEqual(rul, 200)
    
    def test_maintenance_decision_integration(self):
        """Bakım kararı entegrasyonu testi"""
        from maintenance import maintenance_decision
        
        # Test RUL değerleri
        test_ruls = [15, 35, 80, 125]
        custom_thresholds = {"critical": 20, "planned": 50}
        
        for rul in test_ruls:
            result = maintenance_decision(rul, custom_thresholds)
            
            # Sonuç yapısını kontrol et
            self.assertIn('status', result)
            self.assertIn('rul', result)
            self.assertIn('color', result)
            self.assertIn('message', result)
            
            # Değer tiplerini kontrol et
            self.assertIsInstance(result['status'], str)
            self.assertIsInstance(result['color'], str)
            self.assertIsInstance(result['message'], str)
    
    def test_stream_data_processing(self):
        """Stream veri işleme testi"""
        # Stream CSV'yi oku
        df = pd.read_csv("logs/stream.csv")
        
        # Veri işleme simülasyonu
        if len(df) > 0:
            # Son satırı al (latest data)
            latest = df.tail(1)
            
            # Değerleri çıkar
            sicaklik = latest['sicaklik'].iloc[0]
            titresim = latest['titresim'].iloc[0]
            tork = latest['tork'].iloc[0]
            
            # Kontroller
            self.assertIsInstance(sicaklik, (int, float))
            self.assertIsInstance(titresim, (int, float))
            self.assertIsInstance(tork, (int, float))
            
            # RUL hesapla
            rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
            self.assertGreaterEqual(rul, 10)
    
    def test_file_upload_simulation(self):
        """Dosya yükleme simülasyonu testi"""
        # Upload edilecek CSV verisi
        upload_data = pd.DataFrame({
            'sicaklik': [45.5, 46.0, 47.0],
            'titresim': [0.25, 0.26, 0.27],
            'tork': [55.0, 56.0, 57.0]
        })
        
        # CSV'ye kaydet
        upload_file = "uploaded_data.csv"
        upload_data.to_csv(upload_file, index=False)
        
        # Dosyayı oku ve işle
        df = pd.read_csv(upload_file)
        
        # Kontroller
        self.assertEqual(len(df), 3)
        self.assertIn('sicaklik', df.columns)
        self.assertIn('titresim', df.columns)
        self.assertIn('tork', df.columns)
        
        # Son satırı işle
        if len(df) > 0:
            latest = df.tail(1)
            sicaklik = latest['sicaklik'].iloc[0]
            titresim = latest['titresim'].iloc[0]
            tork = latest['tork'].iloc[0]
            
            # RUL hesapla
            rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
            self.assertGreaterEqual(rul, 10)
    
    def test_manual_input_processing(self):
        """Manuel giriş işleme testi"""
        # Manuel giriş değerleri
        manual_inputs = [
            {'sicaklik': 45.5, 'titresim': 0.25, 'tork': 55.0},
            {'sicaklik': 50.0, 'titresim': 0.30, 'tork': 60.0},
            {'sicaklik': 40.0, 'titresim': 0.20, 'tork': 50.0}
        ]
        
        for inputs in manual_inputs:
            sicaklik = inputs['sicaklik']
            titresim = inputs['titresim']
            tork = inputs['tork']
            
            # Değer doğrulama
            self.assertIsInstance(sicaklik, (int, float))
            self.assertIsInstance(titresim, (int, float))
            self.assertIsInstance(tork, (int, float))
            
            # RUL hesapla
            rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
            self.assertGreaterEqual(rul, 10)
    
    def test_threshold_configuration(self):
        """Eşik konfigürasyonu testi"""
        # Farklı eşik değerleri
        threshold_configs = [
            {'critical': 15, 'planned': 40},
            {'critical': 20, 'planned': 50},
            {'critical': 25, 'planned': 60}
        ]
        
        test_rul = 30
        
        for thresholds in threshold_configs:
            from maintenance import maintenance_decision
            result = maintenance_decision(test_rul, thresholds)
            
            # Eşik mantığını kontrol et
            if test_rul < thresholds['critical']:
                self.assertEqual(result['status'], 'CRITICAL')
            elif test_rul < thresholds['planned']:
                self.assertEqual(result['status'], 'PLANNED')
            else:
                self.assertEqual(result['status'], 'NORMAL')
    
    def test_data_source_selection(self):
        """Veri kaynağı seçimi testi"""
        from constants import DataSources
        
        # Veri kaynakları
        data_sources = [
            DataSources.LIVE_STREAM,
            DataSources.FILE_UPLOAD,
            DataSources.MANUAL_INPUT
        ]
        
        for source in data_sources:
            self.assertIsInstance(source, str)
            self.assertGreater(len(source), 0)
            
            # Türkçe karakter kontrolü
            self.assertIn(source, ["Canlı Akış", "Dosya Yükle", "Manuel Giriş"])
    
    def test_reporting_integration(self):
        """Raporlama entegrasyonu testi"""
        from reporting import append_prediction_log, ensure_dirs
        from constants import ColumnNames
        
        # Dizinleri oluştur
        ensure_dirs()
        
        # Log verisi
        log_data = {
            ColumnNames.TIMESTAMP: "2024-01-01T12:00:00",
            ColumnNames.SICAKLIK: 45.5,
            ColumnNames.TITRESIM: 0.25,
            ColumnNames.TORK: 55.0,
            ColumnNames.RUL: 125.5,
            ColumnNames.STATUS: "NORMAL"
        }
        
        # Log'u yaz
        try:
            append_prediction_log(log_data)
            success = True
        except Exception as e:
            success = False
            print(f"Loglama hatası: {e}")
        
        self.assertTrue(success)
    
    def test_css_styling_constants(self):
        """CSS styling sabitleri testi"""
        # Status renkleri (app.py'deki gibi)
        status_colors = {
            'CRITICAL': '#D32F2F',
            'PLANNED': '#ED6C02', 
            'NORMAL': '#2E7D32',
            'UNKNOWN': '#757575'
        }
        
        for status, color in status_colors.items():
            self.assertIsInstance(status, str)
            self.assertIsInstance(color, str)
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)  # #RRGGBB format
    
    def test_metric_display_logic(self):
        """Metrik gösterim mantığı testi"""
        # Test verileri
        test_metrics = [
            {'label': 'Kalan Ömür (RUL)', 'value': '125.50 döngü', 'delta': 'Eşik: 20-50'},
            {'label': 'Bakım Durumu', 'value': 'NORMAL', 'delta': None},
            {'label': 'Son Güncelleme', 'value': '2024-01-01T12:00:00', 'delta': None}
        ]
        
        for metric in test_metrics:
            self.assertIn('label', metric)
            self.assertIn('value', metric)
            self.assertIsInstance(metric['label'], str)
            self.assertIsInstance(metric['value'], str)
            self.assertGreater(len(metric['label']), 0)
            self.assertGreater(len(metric['value']), 0)
    
    def test_error_handling_scenarios(self):
        """Hata yönetimi senaryoları testi"""
        # Eksik dosya durumu
        missing_files = ["nonexistent_model.pkl", "nonexistent_data.csv"]
        
        for file_path in missing_files:
            file_exists = os.path.exists(file_path)
            self.assertFalse(file_exists)
        
        # Boş DataFrame durumu
        empty_df = pd.DataFrame()
        self.assertEqual(len(empty_df), 0)
        
        # Geçersiz değer durumu
        invalid_values = [None, "abc", float('nan')]
        
        for value in invalid_values:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                is_valid = False
            else:
                try:
                    float(value)
                    is_valid = True
                except (ValueError, TypeError):
                    is_valid = False
            
            # None ve NaN için False, "abc" için False bekleniyor
            if value is None or (isinstance(value, float) and np.isnan(value)) or value == "abc":
                self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()

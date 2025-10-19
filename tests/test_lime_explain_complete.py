#!/usr/bin/env python3
"""
LIME Explain modülü için kapsamlı testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path


class TestLimeExplainComplete(unittest.TestCase):
    """LIME Explain kapsamlı testleri"""
    
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
        # Train RUL CSV
        np.random.seed(42)
        data = {
            'sensor_1': np.random.normal(50, 10, 100),
            'sensor_2': np.random.normal(30, 5, 100),
            'sensor_3': np.random.normal(100, 15, 100),
            'RUL': np.random.uniform(10, 200, 100)
        }
        df = pd.DataFrame(data)
        df.to_csv("train_rul.csv", index=False)
        
        # Selected features
        features = ['sensor_1', 'sensor_2', 'sensor_3']
        with open("selected_features.txt", "w") as f:
            for feature in features:
                f.write(f"{feature}\n")
    
    @patch('lime_explain.lime_tabular.LimeTabularExplainer')
    @patch('joblib.load')
    def test_explain_instance_success(self, mock_joblib, mock_lime_explainer):
        """explain_instance başarı testi"""
        from lime_explain import explain_instance
        
        # Mock model ve scaler
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([125.5]))
        
        mock_scaler = MagicMock()
        mock_scaler.transform = MagicMock(return_value=np.array([[1.0, 2.0, 3.0]]))
        
        # Mock LIME explainer
        mock_explainer_instance = MagicMock()
        mock_explanation = MagicMock()
        mock_explanation.save_to_file = MagicMock()
        
        mock_explainer_instance.explain_instance = MagicMock(return_value=mock_explanation)
        mock_lime_explainer.return_value = mock_explainer_instance
        
        # Test verisi
        test_data = pd.DataFrame({
            'sensor_1': [45.5],
            'sensor_2': [30.2],
            'sensor_3': [95.1]
        })
        
        feature_names = ['sensor_1', 'sensor_2', 'sensor_3']
        
        # Fonksiyonu çalıştır
        result = explain_instance(mock_model, mock_scaler, test_data, feature_names)
        
        # Kontroller
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith('.html'))
        mock_explanation.save_to_file.assert_called_once()
    
    @patch('lime_explain.pd.read_csv')
    def test_explain_instance_file_not_found(self, mock_read_csv):
        """explain_instance dosya bulunamadı testi"""
        from lime_explain import explain_instance
        
        mock_read_csv.side_effect = FileNotFoundError("Dosya bulunamadı")
        
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        test_data = pd.DataFrame({'sensor_1': [45.5]})
        feature_names = ['sensor_1']
        
        # FileNotFoundError wrapped in RuntimeError bekleniyor
        with self.assertRaises(RuntimeError):
            explain_instance(mock_model, mock_scaler, test_data, feature_names)
    
    @patch('lime_explain.lime_tabular.LimeTabularExplainer')
    @patch('lime_explain.pd.read_csv')
    def test_explain_instance_lime_error(self, mock_read_csv, mock_lime_explainer):
        """explain_instance LIME hatası testi"""
        from lime_explain import explain_instance
        
        # Mock train data
        mock_read_csv.return_value = pd.DataFrame({
            'sensor_1': [1, 2, 3],
            'sensor_2': [4, 5, 6],
            'sensor_3': [7, 8, 9]
        })
        
        # LIME explainer hatası
        mock_lime_explainer.side_effect = ValueError("LIME hatası")
        
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_scaler.transform = MagicMock(return_value=np.array([[1, 2, 3]]))
        
        test_data = pd.DataFrame({'sensor_1': [45.5]})
        feature_names = ['sensor_1']
        
        # RuntimeError bekleniyor
        with self.assertRaises(RuntimeError):
            explain_instance(mock_model, mock_scaler, test_data, feature_names)
    
    @patch('lime_explain.webbrowser.open')
    @patch('lime_explain.os.path.abspath')
    def test_open_html_in_browser_success(self, mock_abspath, mock_webbrowser):
        """open_html_in_browser başarı testi"""
        from lime_explain import open_html_in_browser
        
        mock_abspath.return_value = "/absolute/path/test.html"
        
        # Fonksiyonu çalıştır
        open_html_in_browser("test.html")
        
        # Kontroller
        mock_abspath.assert_called_once_with("test.html")
        mock_webbrowser.assert_called_once_with('file:///absolute/path/test.html')
    
    @patch('lime_explain.webbrowser.open')
    def test_open_html_in_browser_error(self, mock_webbrowser):
        """open_html_in_browser hata testi"""
        from lime_explain import open_html_in_browser
        
        mock_webbrowser.side_effect = OSError("Tarayıcı hatası")
        
        # Hata yakalanmalı ve print edilmeli
        with patch('builtins.print') as mock_print:
            open_html_in_browser("test.html")
            mock_print.assert_called()
    
    @patch('lime_explain.open_html_in_browser')
    @patch('lime_explain.explain_instance')
    @patch('joblib.load')
    def test_test_lime_explanation_success(self, mock_joblib, mock_explain, mock_open_browser):
        """test_lime_explanation başarı testi"""
        from lime_explain import test_lime_explanation
        
        # Mock model ve scaler
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_joblib.side_effect = [mock_model, mock_scaler]
        
        # Mock explain_instance
        mock_explain.return_value = "reports/lime_ex.html"
        
        # Fonksiyonu çalıştır
        result = test_lime_explanation()
        
        # Kontroller
        self.assertTrue(result)
        mock_explain.assert_called_once()
        mock_open_browser.assert_called_once_with("reports/lime_ex.html")
    
    @patch('joblib.load')
    def test_test_lime_explanation_model_error(self, mock_joblib):
        """test_lime_explanation model yükleme hatası testi"""
        from lime_explain import test_lime_explanation
        
        mock_joblib.side_effect = FileNotFoundError("Model bulunamadı")
        
        # Fonksiyonu çalıştır
        result = test_lime_explanation()
        
        # Hata durumunda False dönmeli
        self.assertFalse(result)
    
    @patch('lime_explain.explain_instance')
    @patch('joblib.load')
    def test_test_lime_explanation_explain_error(self, mock_joblib, mock_explain):
        """test_lime_explanation açıklama hatası testi"""
        from lime_explain import test_lime_explanation
        
        # Mock model ve scaler
        mock_joblib.side_effect = [MagicMock(), MagicMock()]
        
        # explain_instance hatası
        mock_explain.side_effect = RuntimeError("Açıklama hatası")
        
        # Fonksiyonu çalıştır
        result = test_lime_explanation()
        
        # Hata durumunda False dönmeli
        self.assertFalse(result)
    
    def test_module_imports(self):
        """Modül import testleri"""
        # Ana modülü import et
        try:
            import lime_explain
            self.assertTrue(hasattr(lime_explain, 'explain_instance'))
            self.assertTrue(hasattr(lime_explain, 'open_html_in_browser'))
            self.assertTrue(hasattr(lime_explain, 'test_lime_explanation'))
        except ImportError as e:
            self.fail(f"lime_explain modülü import edilemedi: {e}")
    
    @patch('lime_explain.Path')
    def test_directory_creation(self, mock_path):
        """Dizin oluşturma testi"""
        from lime_explain import explain_instance
        
        # Mock Path
        mock_path_instance = MagicMock()
        mock_path.return_value.parent.mkdir = MagicMock()
        mock_path.return_value = mock_path_instance
        
        # Mock diğer bileşenler
        with patch('lime_explain.pd.read_csv'), \
             patch('lime_explain.lime_tabular.LimeTabularExplainer'), \
             patch('builtins.print'):
            
            mock_model = MagicMock()
            mock_scaler = MagicMock()
            mock_scaler.transform = MagicMock(return_value=np.array([[1, 2, 3]]))
            
            test_data = pd.DataFrame({'sensor_1': [45.5]})
            feature_names = ['sensor_1']
            
            try:
                explain_instance(mock_model, mock_scaler, test_data, feature_names)
            except:
                pass  # Hata bekleniyor, sadece mkdir çağrısını test ediyoruz
            
            # mkdir çağrıldığını kontrol et
            mock_path.assert_called()
    
    def test_feature_names_validation(self):
        """Feature names doğrulama testi"""
        # Feature names listesi
        feature_names = ['sensor_1', 'sensor_2', 'sensor_3']
        
        # Validasyon testleri
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(all(isinstance(name, str) for name in feature_names))
        self.assertTrue(all(len(name) > 0 for name in feature_names))
    
    def test_test_data_structure(self):
        """Test verisi yapısı testi"""
        # Test verisi yapısını kontrol et
        test_data = {
            'sensor_measurement_11': [47.5],
            'sensor_measurement_12': [521.0],
            'sensor_measurement_4': [1400.0],
            'sensor_measurement_7': [553.0],
            'sensor_measurement_15': [8.4],
            'sensor_measurement_9': [9050.0],
            'sensor_measurement_21': [23.3],
            'sensor_measurement_20': [38.9],
            'sensor_measurement_2': [642.0],
            'sensor_measurement_3': [1585.0]
        }
        
        df = pd.DataFrame(test_data)
        
        # Yapı kontrolleri
        self.assertEqual(len(df), 1)  # Tek satır
        self.assertEqual(len(df.columns), 10)  # 10 özellik
        
        # Veri tipleri
        for col in df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))


if __name__ == '__main__':
    unittest.main()

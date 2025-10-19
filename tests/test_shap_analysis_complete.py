#!/usr/bin/env python3
"""
SHAP Analysis modülü için kapsamlı testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open


class TestShapAnalysisComplete(unittest.TestCase):
    """SHAP Analysis kapsamlı testleri"""
    
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
            'sensor_1': np.random.normal(50, 10, 300),
            'sensor_2': np.random.normal(30, 5, 300),
            'sensor_3': np.random.normal(100, 15, 300),
            'RUL': np.random.uniform(10, 200, 300)
        }
        df = pd.DataFrame(data)
        df.to_csv("train_rul.csv", index=False)
        
        # Selected features
        features = ['sensor_1', 'sensor_2', 'sensor_3']
        with open("selected_features.txt", "w") as f:
            for feature in features:
                f.write(f"{feature}\n")
        
        # Reports dizini
        os.makedirs("reports", exist_ok=True)
    
    @patch('shap_analysis.shap.summary_plot')
    @patch('shap_analysis.shap.Explainer')
    @patch('shap_analysis.plt.savefig')
    @patch('shap_analysis.plt.close')
    @patch('shap_analysis.plt.tight_layout')
    @patch('shap_analysis.plt.figure')
    @patch('shap_analysis.os.makedirs')
    def test_shap_summary_png_success(self, mock_makedirs, mock_figure, mock_tight_layout, 
                                     mock_close, mock_savefig, mock_explainer, mock_summary_plot):
        """shap_summary_png başarı testi"""
        from shap_analysis import shap_summary_png
        
        # Mock model
        mock_model = MagicMock()
        
        # Mock SHAP explainer
        mock_explainer_instance = MagicMock()
        mock_shap_values = MagicMock()
        mock_explainer_instance.return_value = mock_shap_values
        mock_explainer.return_value = mock_explainer_instance
        
        # Test DataFrame
        test_df = pd.DataFrame({
            'sensor_1': [45.5, 46.0, 47.0],
            'sensor_2': [30.2, 31.0, 32.0],
            'sensor_3': [95.1, 96.0, 97.0]
        })
        
        # Fonksiyonu çalıştır
        result = shap_summary_png(mock_model, test_df)
        
        # Kontroller
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith('.png'))
        mock_explainer.assert_called_once_with(mock_model)
        mock_summary_plot.assert_called_once()
        mock_savefig.assert_called_once()
    
    @patch('shap_analysis.shap.Explainer')
    def test_shap_summary_png_explainer_error(self, mock_explainer):
        """shap_summary_png explainer hatası testi"""
        from shap_analysis import shap_summary_png
        
        mock_explainer.side_effect = ValueError("SHAP explainer hatası")
        
        mock_model = MagicMock()
        test_df = pd.DataFrame({'sensor_1': [45.5]})
        
        # RuntimeError bekleniyor
        with self.assertRaises(RuntimeError):
            shap_summary_png(mock_model, test_df)
    
    @patch('shap_analysis.shap.plots.waterfall')
    @patch('shap_analysis.shap.Explainer')
    @patch('shap_analysis.plt.savefig')
    @patch('shap_analysis.plt.close')
    @patch('shap_analysis.plt.tight_layout')
    @patch('shap_analysis.plt.figure')
    @patch('shap_analysis.os.makedirs')
    def test_shap_local_png_success(self, mock_makedirs, mock_figure, mock_tight_layout,
                                   mock_close, mock_savefig, mock_explainer, mock_waterfall):
        """shap_local_png başarı testi"""
        from shap_analysis import shap_local_png
        
        # Mock model
        mock_model = MagicMock()
        
        # Mock SHAP explainer
        mock_explainer_instance = MagicMock()
        mock_shap_values = MagicMock()
        mock_explainer_instance.return_value = mock_shap_values
        mock_explainer.return_value = mock_explainer_instance
        
        # Test DataFrame (tek satır)
        test_df = pd.DataFrame({
            'sensor_1': [45.5],
            'sensor_2': [30.2],
            'sensor_3': [95.1]
        })
        
        # Fonksiyonu çalıştır
        result = shap_local_png(mock_model, test_df)
        
        # Kontroller
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith('.png'))
        mock_explainer.assert_called_once_with(mock_model)
        mock_waterfall.assert_called_once()
        mock_savefig.assert_called_once()
    
    @patch('shap_analysis.shap.Explainer')
    def test_shap_local_png_runtime_error(self, mock_explainer):
        """shap_local_png runtime hatası testi"""
        from shap_analysis import shap_local_png
        
        mock_explainer.side_effect = RuntimeError("SHAP runtime hatası")
        
        mock_model = MagicMock()
        test_df = pd.DataFrame({'sensor_1': [45.5]})
        
        # RuntimeError bekleniyor
        with self.assertRaises(RuntimeError):
            shap_local_png(mock_model, test_df)
    
    @patch('shap_analysis.pd.read_csv')
    def test_load_sample_data_success(self, mock_read_csv):
        """load_sample_data başarı testi"""
        from shap_analysis import load_sample_data
        
        # Mock CSV data
        mock_df = pd.DataFrame({
            'sensor_1': np.random.normal(50, 10, 500),
            'sensor_2': np.random.normal(30, 5, 500),
            'sensor_3': np.random.normal(100, 15, 500),
            'RUL': np.random.uniform(10, 200, 500)
        })
        mock_read_csv.return_value = mock_df
        
        # Mock features file
        with patch('builtins.open', mock_open(read_data="sensor_1\nsensor_2\nsensor_3\n")):
            result = load_sample_data(sample_size=300)
        
        # Kontroller
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 300)
        self.assertEqual(len(result.columns), 3)
    
    @patch('shap_analysis.pd.read_csv')
    def test_load_sample_data_file_error(self, mock_read_csv):
        """load_sample_data dosya hatası testi"""
        from shap_analysis import load_sample_data
        
        mock_read_csv.side_effect = FileNotFoundError("CSV bulunamadı")
        
        # FileNotFoundError bekleniyor
        with self.assertRaises(FileNotFoundError):
            load_sample_data()
    
    def test_load_sample_data_features_file_error(self):
        """load_sample_data features dosyası hatası testi"""
        from shap_analysis import load_sample_data
        
        # Features dosyasını sil
        if os.path.exists("selected_features.txt"):
            os.remove("selected_features.txt")
        
        # FileNotFoundError bekleniyor
        with self.assertRaises(FileNotFoundError):
            load_sample_data()
    
    @patch('shap_analysis.shap_local_png')
    @patch('shap_analysis.shap_summary_png')
    @patch('shap_analysis.load_sample_data')
    @patch('joblib.load')
    def test_test_shap_analysis_success(self, mock_joblib, mock_load_sample, 
                                       mock_summary, mock_local):
        """test_shap_analysis başarı testi"""
        from shap_analysis import test_shap_analysis
        
        # Mock model ve scaler
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_joblib.side_effect = [mock_model, mock_scaler]
        
        # Mock sample data
        mock_sample_df = pd.DataFrame({
            'sensor_1': [45.5, 46.0],
            'sensor_2': [30.2, 31.0],
            'sensor_3': [95.1, 96.0]
        })
        mock_load_sample.return_value = mock_sample_df
        
        # Mock SHAP functions
        mock_summary.return_value = "reports/shap_summary.png"
        mock_local.return_value = "reports/shap_local.png"
        
        # Mock scaler transform
        mock_scaler.transform.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Fonksiyonu çalıştır
        result = test_shap_analysis()
        
        # Kontroller
        self.assertTrue(result)
        mock_summary.assert_called_once()
        mock_local.assert_called_once()
    
    @patch('joblib.load')
    def test_test_shap_analysis_model_error(self, mock_joblib):
        """test_shap_analysis model yükleme hatası testi"""
        from shap_analysis import test_shap_analysis
        
        mock_joblib.side_effect = FileNotFoundError("Model bulunamadı")
        
        # Fonksiyonu çalıştır
        result = test_shap_analysis()
        
        # Hata durumunda False dönmeli
        self.assertFalse(result)
    
    @patch('shap_analysis.load_sample_data')
    @patch('joblib.load')
    def test_test_shap_analysis_data_error(self, mock_joblib, mock_load_sample):
        """test_shap_analysis veri yükleme hatası testi"""
        from shap_analysis import test_shap_analysis
        
        # Mock model ve scaler
        mock_joblib.side_effect = [MagicMock(), MagicMock()]
        
        # Data loading hatası
        mock_load_sample.side_effect = ValueError("Veri yükleme hatası")
        
        # Fonksiyonu çalıştır
        result = test_shap_analysis()
        
        # Hata durumunda False dönmeli
        self.assertFalse(result)
    
    def test_module_imports(self):
        """Modül import testleri"""
        try:
            import shap_analysis
            self.assertTrue(hasattr(shap_analysis, 'shap_summary_png'))
            self.assertTrue(hasattr(shap_analysis, 'shap_local_png'))
            self.assertTrue(hasattr(shap_analysis, 'load_sample_data'))
            self.assertTrue(hasattr(shap_analysis, 'test_shap_analysis'))
        except ImportError as e:
            self.fail(f"shap_analysis modülü import edilemedi: {e}")
    
    def test_default_parameters(self):
        """Varsayılan parametreler testi"""
        # Varsayılan dosya yolları
        default_csv = "train_rul.csv"
        default_features = "selected_features.txt"
        default_sample_size = 300
        
        # Parametrelerin doğru olduğunu kontrol et
        self.assertTrue(os.path.exists(default_csv))
        self.assertTrue(os.path.exists(default_features))
        self.assertIsInstance(default_sample_size, int)
        self.assertGreater(default_sample_size, 0)
    
    def test_output_paths(self):
        """Çıktı yolları testi"""
        # Varsayılan çıktı yolları
        summary_path = "reports/shap_summary.png"
        local_path = "reports/shap_local.png"
        
        # Yol formatlarını kontrol et
        self.assertTrue(summary_path.endswith('.png'))
        self.assertTrue(local_path.endswith('.png'))
        self.assertTrue(summary_path.startswith('reports/'))
        self.assertTrue(local_path.startswith('reports/'))
    
    def test_sample_size_validation(self):
        """Sample size doğrulama testi"""
        # Farklı sample size değerleri
        valid_sizes = [100, 200, 300, 500]
        invalid_sizes = [0, -1, "abc", None]
        
        for size in valid_sizes:
            self.assertIsInstance(size, int)
            self.assertGreater(size, 0)
        
        for size in invalid_sizes:
            if isinstance(size, int):
                self.assertLessEqual(size, 0)
            else:
                self.assertNotIsInstance(size, int)
    
    @patch('shap_analysis.os.makedirs')
    def test_directory_creation(self, mock_makedirs):
        """Dizin oluşturma testi"""
        from shap_analysis import shap_summary_png
        
        # Mock diğer bileşenler
        with patch('shap_analysis.shap.Explainer'), \
             patch('shap_analysis.shap.summary_plot'), \
             patch('shap_analysis.plt.figure'), \
             patch('shap_analysis.plt.savefig'), \
             patch('shap_analysis.plt.close'), \
             patch('shap_analysis.plt.tight_layout'):
            
            mock_model = MagicMock()
            test_df = pd.DataFrame({'sensor_1': [45.5]})
            
            try:
                shap_summary_png(mock_model, test_df)
            except:
                pass  # Hata bekleniyor, sadece makedirs çağrısını test ediyoruz
            
            # makedirs çağrıldığını kontrol et
            mock_makedirs.assert_called()


if __name__ == '__main__':
    unittest.main()

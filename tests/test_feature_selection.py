#!/usr/bin/env python3
"""
Feature Selection modülü için unit testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from feature_selection import main
from constants import FilePaths


class TestFeatureSelection(unittest.TestCase):
    """Feature Selection testleri"""
    
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
        """Test için örnek veri oluştur"""
        # Örnek RUL verisi
        np.random.seed(42)
        n_samples = 100
        n_features = 25
        
        data = {
            'unit_number': np.repeat(range(1, 6), 20),
            'time_in_cycles': np.tile(range(1, 21), 5),
            'RUL': np.random.uniform(10, 200, n_samples)
        }
        
        # Sensör verileri ekle
        for i in range(1, n_features):
            data[f'sensor_measurement_{i}'] = np.random.normal(50, 10, n_samples)
        
        # Operational settings
        for i in range(1, 4):
            data[f'op_setting_{i}'] = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame(data)
        df.to_csv(FilePaths.TRAIN_RUL_CSV, index=False)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.barh')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.figure')
    def test_main_function_success(self, mock_figure, mock_title, mock_barh, 
                                  mock_tight_layout, mock_close, mock_savefig):
        """Ana fonksiyon başarı testi"""
        
        # Mock matplotlib
        mock_figure.return_value = MagicMock()
        
        # Ana fonksiyonu çalıştır
        try:
            main()
            success = True
        except Exception as e:
            success = False
            print(f"Hata: {e}")
        
        self.assertTrue(success)
        
        # Çıktı dosyalarının oluştuğunu kontrol et
        self.assertTrue(os.path.exists("selected_features.txt"))
        self.assertTrue(os.path.exists("feature_importance.png"))
        
        # Selected features dosyasını kontrol et
        with open("selected_features.txt", "r") as f:
            features = f.read().strip().split('\n')
        
        self.assertEqual(len(features), 10)  # TOP_K = 10
        self.assertTrue(all(len(f.strip()) > 0 for f in features))
    
    def test_missing_input_file(self):
        """Eksik input dosyası testi"""
        # CSV dosyasını sil
        if os.path.exists(FilePaths.TRAIN_RUL_CSV):
            os.remove(FilePaths.TRAIN_RUL_CSV)
        
        # FileNotFoundError fırlatmalı
        with self.assertRaises(FileNotFoundError):
            main()
    
    @patch('pandas.read_csv')
    def test_empty_dataframe(self, mock_read_csv):
        """Boş DataFrame testi"""
        # Boş DataFrame döndür
        mock_read_csv.return_value = pd.DataFrame()
        
        # Hata fırlatmalı
        with self.assertRaises((KeyError, ValueError)):
            main()
    
    @patch('pandas.read_csv')
    def test_missing_rul_column(self, mock_read_csv):
        """RUL kolonu eksik testi"""
        # RUL kolonu olmayan DataFrame
        df = pd.DataFrame({
            'unit_number': [1, 2, 3],
            'time_in_cycles': [1, 2, 3],
            'sensor_1': [1.0, 2.0, 3.0]
        })
        mock_read_csv.return_value = df
        
        # KeyError fırlatmalı
        with self.assertRaises(KeyError):
            main()
    
    @patch('sklearn.ensemble.RandomForestRegressor')
    @patch('pandas.read_csv')
    def test_random_forest_training(self, mock_read_csv, mock_rf):
        """RandomForest eğitim testi"""
        # Mock DataFrame
        df = pd.DataFrame({
            'unit_number': [1, 2, 3, 4, 5],
            'time_in_cycles': [1, 2, 3, 4, 5],
            'RUL': [100, 80, 60, 40, 20],
            'sensor_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'sensor_2': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        mock_read_csv.return_value = df
        
        # Mock RandomForest
        mock_rf_instance = MagicMock()
        mock_rf_instance.feature_importances_ = np.array([0.6, 0.4])
        mock_rf.return_value = mock_rf_instance
        
        # Mock diğer bileşenler
        with patch('sklearn.feature_selection.mutual_info_regression') as mock_mi, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.barh'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.figure'):
            
            mock_mi.return_value = np.array([0.5, 0.3])
            
            try:
                main()
                success = True
            except Exception as e:
                success = False
                print(f"Hata: {e}")
            
            self.assertTrue(success)
            mock_rf_instance.fit.assert_called_once()
    
    def test_feature_selection_logic(self):
        """Özellik seçim mantığı testi"""
        # Gerçek veri ile test
        try:
            main()
            
            # Seçilen özellikleri oku
            with open("selected_features.txt", "r") as f:
                selected_features = [line.strip() for line in f.readlines()]
            
            # Kontroller
            self.assertEqual(len(selected_features), 10)
            self.assertTrue(all(isinstance(f, str) for f in selected_features))
            self.assertTrue(all(len(f) > 0 for f in selected_features))
            
            # Tekrar eden özellik olmamalı
            self.assertEqual(len(selected_features), len(set(selected_features)))
            
        except Exception as e:
            self.fail(f"Feature selection failed: {e}")
    
    @patch('builtins.print')
    def test_output_messages(self, mock_print):
        """Çıktı mesajları testi"""
        try:
            main()
            
            # Print çağrılarını kontrol et
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            
            # Beklenen mesajları ara
            found_top_features = any("En önemli özellikler" in str(call) for call in print_calls)
            found_saved_message = any("kaydedildi" in str(call) for call in print_calls)
            
            self.assertTrue(found_top_features or found_saved_message)
            
        except Exception as e:
            self.fail(f"Output test failed: {e}")


if __name__ == '__main__':
    unittest.main()
